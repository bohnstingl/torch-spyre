# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: allow-untyped-defs
"""Prototype frontend for `for_each_tile` (torch-spyre#3965, working-set reduction).

`for_each_tile` is `scan` with the tiling made explicit: ONE co-indexed loop level
that reduces every operand to a per-step tile -- a `narrow` view, a whole
invariant, or a gathered pool row -- threads an optional carry, and optionally
lays each step's result tile back into a full-size output along one axis.

The destination of that lay-back is the caller's, via `output=`. `scan` grew an
`out=` parameter for it, so the tile write goes straight into the buffer the caller
already owns: no stacked scratch, and no post-loop fold. Without `output=` the loop
falls back to `scan`'s own stacked buffer, which costs a full-size copy of the result
for any `out_dim != 0` (`_stacked_to_full`) -- the frontend cannot avoid it, because
it does not learn the output tile's shape until `body` has run and so has nothing to
allocate.
"""

import contextlib
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Sequence

import torch
import torch._prims_common as utils
from torch._higher_order_ops.scan import scan


__all__ = ["Gather", "for_each_tile"]


@dataclass(frozen=True)
class Gather:
    """Tile spec for a POOLED operand: step `i` sees `pool.index_select(axis, index[i])`.

    `index` is a contiguous 1-D integer tensor and its length corresponds to the
    loop's trip count.

    As an `out_dim` it is the write dual -- step `i` does
    `output.index_copy_(axis, index[i], y_i)` -- and then `index` may also be
    `[num_tiles, w]`, one row of `w` destination positions per step.
    """

    axis: int
    index: torch.Tensor


class Kind(enum.Enum):
    SLICE = "slice"
    GATHER = "gather"
    INVARIANT = "invariant"


@dataclass(frozen=True)
class TileSpec:
    """How one operand is reduced to its per-step tile.

    `kind` picks the reduction: SLICE narrows the operand, GATHER indexes a pool with
    `index`, INVARIANT passes it whole.

    `dim` is the tiled axis for SLICE and the pool axis for GATHER.

    `tile_size` is the tile's shape: a per-dimension vector of the operand's own rank,
    equal to the operand's shape except on the tiled axis. The notation is aimed to be
    similar to the one used in the sibling `tile.py`.
    Note: tiles are RANK-PRESERVING (`narrow`, not `select`).

    `num_tiles` is the operand's own tile count: `shape[dim] // tile_size[dim]` for
    SLICE, `len(index)` for GATHER, 1 for an INVARIANT operand, which is not tiled at
    all.

    `index` is the GATHER index table, `None` for the other kinds.
    """

    kind: Kind
    dim: int = 0
    tile_size: tuple[int, ...] = ()
    num_tiles: int = 1
    index: torch.Tensor | None = None

    @property
    def extent(self) -> int:
        """The tile's size along the tiled axis."""
        return self.tile_size[self.dim] if self.tile_size else 1


DimSpec = Union[int, None, Gather]  # noqa: UP007  # `int | None | Gather` needs Gather at runtime


def _tile_size_vector(shape, dim: int, extent: int) -> tuple[int, ...]:
    """The operand's shape with the tiled axis narrowed to `extent`."""
    tile_size = list(shape)
    tile_size[dim] = extent
    return tuple(tile_size)


def _normalize_in_specs(operands, dims, tile_size: int) -> tuple[list[TileSpec], int]:
    """Resolve the per-operand specs and the loop's trip count."""
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError(
            f"for_each_tile() tile_size must be a positive int, got {tile_size!r}"
        )
    if isinstance(dims, list):
        raise TypeError(
            f"for_each_tile() dims must be a tuple, got a list: dims={tuple(dims)!r}"
        )
    per_operand: Sequence[DimSpec] = (
        dims if isinstance(dims, tuple) else (dims,) * len(operands)
    )
    if len(per_operand) != len(operands):
        raise ValueError(
            f"for_each_tile() got {len(per_operand)} dims entries for "
            f"{len(operands)} operands"
        )

    specs: list[TileSpec] = []
    # Every tiled operand is walked by ONE counter, so they must all yield the same
    # number of tiles: x[3, 4] tiled by 2 on dim 1 goes with y[4, 5] tiled by 2 on
    # dim 0, and with a 2-page gather, but not with y[6, 5].
    num_tiles: int | None = None
    num_tiles_of: int | None = None

    for k, (o, d) in enumerate(zip(operands, per_operand)):
        if not isinstance(o, torch.Tensor):
            raise ValueError(f"for_each_tile() operand {k} is not a Tensor: {type(o)}")
        if d is None:
            specs.append(TileSpec(Kind.INVARIANT, tile_size=tuple(o.shape)))
            continue
        if isinstance(d, Gather):
            idx = d.index
            if not isinstance(idx, torch.Tensor) or idx.ndim != 1:
                raise ValueError(
                    f"for_each_tile() Gather index for operand {k} must be a 1-D "
                    f"tensor, got {tuple(idx.shape) if isinstance(idx, torch.Tensor) else type(idx)}"
                )
            if idx.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    f"for_each_tile() Gather index for operand {k} must be int32 or "
                    f"int64, got {idx.dtype}"
                )
            # The frontend will not make the index tensor, e.g., a page table,
            # contiguous itself - it is the responsibility of the user.
            if not idx.is_contiguous():
                raise ValueError(
                    f"for_each_tile() Gather index for operand {k} must be contiguous, "
                    f"got strides {tuple(idx.stride())}; pass index.contiguous()"
                )
            axis = utils.canonicalize_dim(o.ndim, d.axis)
            spec = TileSpec(
                Kind.GATHER,
                axis,
                _tile_size_vector(o.shape, axis, 1),
                idx.shape[0],
                idx,
            )
        elif isinstance(d, int) and not isinstance(d, bool):
            axis = utils.canonicalize_dim(o.ndim, d)
            length = o.shape[axis]
            if length % tile_size != 0:
                raise ValueError(
                    f"for_each_tile() operand {k} has size {length} along dim {d}, "
                    f"which is not a multiple of tile_size={tile_size} (ragged tiles "
                    f"are not supported)"
                )
            spec = TileSpec(
                Kind.SLICE,
                axis,
                _tile_size_vector(o.shape, axis, tile_size),
                length // tile_size,
            )
        else:
            raise ValueError(
                f"for_each_tile() dims entry {k} must be an int, None or Gather, "
                f"got {d!r}"
            )

        if num_tiles is None:
            num_tiles, num_tiles_of = spec.num_tiles, k
        elif spec.num_tiles != num_tiles:
            raise ValueError(
                f"for_each_tile() operands must yield the same number of tiles: "
                f"operand {num_tiles_of} yields {num_tiles}, operand {k} yields "
                f"{spec.num_tiles}"
            )
        specs.append(spec)

    if num_tiles is None:
        raise ValueError(
            "for_each_tile() needs at least one sliced or gathered operand; a "
            "counter-only loop cannot be traced (see torch-spyre#3965, I1)"
        )
    # With no sliced operand nothing reads tile_size: a gathered step takes one pool
    # row, and the trip count comes from the index table. Silently ignoring the
    # argument would let a wrong one look right.
    if tile_size != 1 and all(s.kind is not Kind.SLICE for s in specs):
        raise ValueError(
            f"for_each_tile() got tile_size={tile_size} with no sliced operand; a "
            f"gathered step takes one pool row, so tile_size must be 1"
        )
    return specs, num_tiles


def _movedim(t: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    """Mode the tile dimension to the right place.

    Moving the tiled axis is a pure stride permutation, i.e. a view, so it should
    never materialize any copies.
    """
    return t if src == dst else torch.movedim(t, src, dst)


def _xs_leaf(operand: torch.Tensor, spec: TileSpec) -> torch.Tensor:
    """The scan `xs` leaf for one operand: [num_tiles, ...], scanned along dim 0."""
    if spec.kind is Kind.GATHER:
        if spec.index is None:
            raise AssertionError("GATHER spec without an index table")
        return spec.index
    moved = _movedim(operand, spec.dim, 0)
    # Splitting dim 0 is always expressible in strides, so this is a view.
    return moved.unflatten(0, (moved.shape[0] // spec.extent, spec.extent))


def _tile(operand: torch.Tensor, spec: TileSpec, sliced: torch.Tensor) -> torch.Tensor:
    """Turn scan's step slice of the xs leaf back into the operand's own layout."""
    if spec.kind is Kind.GATHER:
        return operand.index_select(spec.dim, sliced.reshape(1))
    return _movedim(sliced, 0, spec.dim)


def _step_counter(like: torch.Tensor) -> torch.Tensor:
    """The int64 scalar `scan` carry map mode uses when the caller supplies none."""
    return torch.zeros((), dtype=torch.int64, device=like.device)


def _stacked_to_full(ys: torch.Tensor, dim: int) -> torch.Tensor:
    """Fold scan's leading step axis into the tiled output axis.

    `ys` is [num_tiles, *tile_size]; the result is `tile_size` with `dim` grown to
    `num_tiles * extent`. For dim == 0 this is a view, (4, 2, 6) -> (8, 6) on the same
    storage, since the flatten merges two contiguous leading axes. For dim != 0 the
    flatten crosses the moved axis, which strides cannot express, so it copies the whole
    output ((3, 8, 2) -> (8, 6)); `output=` removes that by writing tile i in place.
    """
    return _movedim(ys, 0, dim).flatten(dim, dim + 1)


def _dest_view(output: torch.Tensor, axis: int, num_tiles: int) -> torch.Tensor:
    """`scan`'s `out=` destination for a tiled write: [num_tiles, *tile] over `output`.

    Splitting `axis` into (num_tiles, extent) and moving the tile axis to the front are
    both pure stride permutations, so what `scan` writes through and what the caller
    handed over are the same storage: `out_dim` has become a stride. That is why there
    is nothing to fold afterwards -- compare `_stacked_to_full`, which has to place the
    tiles after the fact and cannot express `out_dim != 0` as strides at all.
    """
    extent = output.shape[axis] // num_tiles
    return _movedim(output.unflatten(axis, (num_tiles, extent)), axis, 0)


def _normalize_out_spec(output, out_dim, num_tiles: int, reverse: bool):
    """Validate `output` against `out_dim` and the trip count.

    Returns the `out_dim=Gather` index table, cast to the dtype `index_copy_` demands,
    or `None` for the sliced and reduction cases.
    """
    if out_dim is None:
        if output is not None:
            raise ValueError(
                "for_each_tile() output= needs out_dim=; a destination the body writes "
                "whole every step is a carry, pass it as init="
            )
        return None

    if output is not None:
        if not isinstance(output, torch.Tensor):
            raise ValueError(
                f"for_each_tile() output must be a Tensor, got {type(output)}"
            )
        if torch.is_grad_enabled():
            # The write is `scan`'s in-place mutation route, which carries no gradient
            # for the destination; dynamo refuses the mutating body outright.
            raise RuntimeError(
                "for_each_tile() output= is inference-only, because a destination "
                "written in place cannot be differentiated. Wrap the call in "
                "torch.no_grad() or torch.inference_mode()."
            )

    if isinstance(out_dim, Gather):
        if output is None:
            raise ValueError(
                "for_each_tile() out_dim=Gather needs output=: a scatter of tiles has "
                "nowhere to go without the destination to scatter into"
            )
        idx = out_dim.index
        if not isinstance(idx, torch.Tensor) or idx.ndim not in (1, 2):
            shape = tuple(idx.shape) if isinstance(idx, torch.Tensor) else type(idx)
            raise ValueError(
                f"for_each_tile() out_dim=Gather index must be a 1-D or 2-D tensor, "
                f"got {shape}"
            )
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"for_each_tile() out_dim=Gather index must be int32 or int64, got "
                f"{idx.dtype}"
            )
        if not idx.is_contiguous():
            raise ValueError(
                f"for_each_tile() out_dim=Gather index must be contiguous, got strides "
                f"{tuple(idx.stride())}; pass index.contiguous()"
            )
        if idx.shape[0] != num_tiles:
            raise ValueError(
                f"for_each_tile() out_dim=Gather index has {idx.shape[0]} rows but the "
                f"loop runs {num_tiles} tiles"
            )
        # `index_copy_` takes an int64 index only, while a Spyre page table is int32.
        # The cast is of the table, not of the data, and only when it is needed.
        return idx if idx.dtype is torch.int64 else idx.to(torch.int64)

    if not isinstance(out_dim, int) or isinstance(out_dim, bool):
        raise ValueError(
            f"for_each_tile() out_dim must be an int, a Gather or None, got {out_dim!r}"
        )
    if output is None:
        return None
    if reverse:
        # The write position is `select(0, step)` of the destination view, so a reversed
        # visit would lay tile i at position num_tiles-1-i. `scan` reverses by flipping,
        # which is a copy and needs a buffer of its own, i.e. exactly what output= is
        # here to avoid.
        raise ValueError("for_each_tile() output= is not supported with reverse=True")
    axis = utils.canonicalize_dim(output.ndim, out_dim)
    if output.shape[axis] % num_tiles != 0:
        raise ValueError(
            f"for_each_tile() output has size {output.shape[axis]} along dim {out_dim}, "
            f"which is not a multiple of the {num_tiles} tiles the loop produces"
        )
    return None


def for_each_tile(
    body,
    operands,
    *,
    dims,
    tile_size: int,
    init=None,
    output=None,
    out_dim=None,
    reverse: bool = False,
):
    """Run `body` once per tile over a co-indexed tiling of `operands`.

    Args:
        body: ``(carry, tiles) -> (next_carry, out_tile)``. ``tiles`` arrives in
            operand order, each operand already reduced to its per-step tile.
            ``next_carry`` is ignored in map mode (``init=None``) and ``out_tile``
            in reduction mode (``out_dim=None``). Same restriction as ``scan``:
            the body may not alias input to output or output to output.
        operands: flat sequence of every input -- sliced, gathered and invariant.
        dims: per-operand tile spec as a tuple, or a single spec broadcast to all of
            them. ``int d`` slices the operand into ``tile_size``-wide contiguous views
            along ``d``; ``None`` passes it whole every step; ``Gather(axis, index)``
            takes one pool row per step.
        tile_size: the tile's size along each tiled axis. The loop's trip count is
            derived: ``shape[dim] // tile_size`` per sliced operand, ``len(index)`` per
            gathered one. All of them must agree.
        init: carry init (tensor or pytree of tensors); ``None`` means no carry.
        output: destination for the result tiles, mutated in place and also returned,
            following the ``out=`` convention. Requires ``out_dim`` and grad mode
            disabled. Omitting it makes the loop allocate a stacked buffer and fold it,
            which costs a full-size copy of the result for ``out_dim != 0``.
        out_dim: ``int d`` lays step ``i``'s tile at ``narrow(d, i*extent, extent)``
            of the output, ``extent = output.shape[d] // num_tiles``.
            ``Gather(axis, index)`` scatters it instead, at
            ``output.index_copy_(axis, index[i], tile)``, and needs ``output``.
            ``None`` means the body emits no tile.
        reverse: visit tiles high to low. The output still lands in natural order.
            Not available together with ``output`` and an ``int`` ``out_dim``.

    Returns:
        ``(final_carry, out)``, either of which is ``None`` for the unused mode. ``out``
        is ``output`` itself whenever that was given.
    """
    operands = tuple(operands)
    specs, num_tiles = _normalize_in_specs(operands, dims, tile_size)
    scatter_index = _normalize_out_spec(output, out_dim, num_tiles, reverse)

    if init is None and out_dim is None:
        raise ValueError("for_each_tile() needs init= (reduction) or out_dim= (map)")

    map_mode = init is None
    if map_mode:
        # scan requires a carry, so map mode (aka using `for_each_tile` similar to `map`)
        # carries the tile counter instead.
        ref = next(o for o, s in zip(operands, specs) if s.kind is not Kind.INVARIANT)
        scan_init = _step_counter(ref)
    else:
        scan_init = init

    xs = tuple(
        _xs_leaf(o, s) for o, s in zip(operands, specs) if s.kind is not Kind.INVARIANT
    )
    if scatter_index is not None:
        # The index table rides as an `xs` leaf, exactly as a gathered operand's does,
        # so the step's destination positions arrive with its tiles.
        xs += (scatter_index,)
        scatter_axis = utils.canonicalize_dim(output.ndim, out_dim.axis)

    def combine_fn(carry, sliced):
        # In map mode, the step counter is the whole carry.
        if map_mode:
            step, carry = carry, None
        it = iter(sliced)
        tiles = tuple(
            o if s.kind is Kind.INVARIANT else _tile(o, s, next(it))
            for o, s in zip(operands, specs)
        )
        next_carry, y = body(carry, tiles)
        if map_mode:
            next_carry = step + 1
        if scatter_index is not None:
            # The scatter is the step's output: `output` reaches the loop as a lifted
            # additional input that the body mutates, which is the one write route
            # `scan` documents for pre-allocated buffers, so nothing is stacked and
            # there is no per-step result to return.
            output.index_copy_(scatter_axis, next(it).reshape(-1), y)
            return next_carry, ()
        return next_carry, (() if out_dim is None else y)

    # specialize_float: a Python float closed over by the body
    # is generalized to a SymFloat and scan_op's validate_subgraph_args_types
    # currently only accepts Tensor, int, SymInt.
    # Note: This is currently a shortcoming of `scan`,
    # but shouldn't be any concern for Spyre.
    ctx = (
        contextlib.nullcontext()
        if torch.compiler.is_dynamo_compiling()
        else torch._dynamo.config.patch(specialize_float=True)
    )
    dest = (
        None
        if output is None or scatter_index is not None
        else _dest_view(output, utils.canonicalize_dim(output.ndim, out_dim), num_tiles)
    )
    with ctx:
        final_carry, ys = scan(
            combine_fn, scan_init, xs, dim=0, reverse=reverse, out=dest
        )

    if out_dim is None:
        return (None if map_mode else final_carry), None
    if output is not None:
        # Written in place, either through `dest` or by the body's own scatter, so `ys`
        # is a view of `output` or empty. Hand back what the caller owns.
        return (None if map_mode else final_carry), output

    if isinstance(ys, (list, tuple)):
        raise ValueError(
            "for_each_tile() out_dim= currently supports a single output tile, "
            f"got a pytree of {len(ys)}"
        )

    axis = utils.canonicalize_dim(ys.ndim - 1, out_dim)
    return (None if map_mode else final_carry), _stacked_to_full(ys, axis)
