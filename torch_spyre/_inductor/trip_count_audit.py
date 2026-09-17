# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Read a spec tree's loop trip count, and audit what may be launched at it.

Rewriting a *concrete* count in a retained tree used to live here too, in two
modes. It no longer does: ``trip_count_symbol.py`` keeps the count symbolic
through the whole frontend instead, so producing another count is one
substitution into ``LoopSpec.count`` and needs none of the extent arithmetic a
rewrite did. The mode that survives as the design -- move the loop bound, leave
every descriptor sized for the maximum -- is now what the tree *is*, not a choice
made after the fact.

What is left is the two read-only questions a caller still has:

* :func:`trip_count_of` -- the single concrete count a tree was built at, for a
  tree the frontend baked (the knob off). It refuses a symbolic count on purpose:
  a symbolic tree has no such count, and answering with the maximum would look
  like a real count to whoever asked.
* :func:`check_prefix_addressable` -- which tiled inputs cannot be fed an
  *oversized* buffer. Sharing one max-size buffer across counts is a claim about
  addressing, independent of anything above, and it is false for some operands
  (paged attention's mask). It is reported rather than raised on, because it is
  the caller's to act on.

Both fail closed: a tree neither can prove it understands raises
:class:`TripCountAuditError`.
"""

import math
from typing import Any, NamedTuple

import sympy

from .op_spec import LoopSpec, OpSpec, TensorArg
from .pass_utils import coeff_through_floor
from .trip_count_symbol import is_symbolic_count


class TripCountAuditError(Exception):
    """This spec tree is not one whose trip count can be read off safely."""


def _iter_loops(specs: list[Any], depth: int = 0):
    """Yield every ``(LoopSpec, depth)`` in ``specs``, outermost first."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            yield entry, depth
            yield from _iter_loops(entry.body, depth + 1)


def _concrete_count(count: Any) -> int:
    """``count`` as a Python int, or raise if it is not a concrete integer."""
    if isinstance(count, bool):
        raise TripCountAuditError(f"loop count is a bool: {count!r}")
    if isinstance(count, int):
        return count
    if isinstance(count, sympy.Integer):
        return int(count)
    raise TripCountAuditError(
        f"loop count {count!r} is symbolic; this tree was traced with the count "
        "left to launch, so it has no single concrete count to report (see "
        "trip_count_symbol.symbolic_count_maxima for what it does have)"
    )


def _count_for_sizing(count: Any) -> int:
    """The count every descriptor in this tree is *sized* for.

    Unlike :func:`_concrete_count` this accepts a symbolic count, answering with
    the maximum it was traced at. That is the right answer for an addressing
    audit specifically: the buffer extents a symbolic tree carries are the
    maximum's, and an oversized-buffer claim is about those extents rather than
    about whatever count is armed at launch.
    """
    if is_symbolic_count(count):
        return int(count.max_value)
    return _concrete_count(count)


def trip_count_of(specs: list[Any]) -> int:
    """Return the single concrete trip count ``specs`` was built at.

    Raises :class:`TripCountAuditError` unless the tree holds exactly one loop
    level with one concrete count. A nested loop is refused rather than guessed
    at: ``OpSpec.tiled_symbols`` is indexed innermost-first, so mapping a level
    to its loop requires knowing the full nesting depth at each op, and nothing
    in the paged-attention case needs it.
    """
    loops = list(_iter_loops(specs))
    if not loops:
        raise TripCountAuditError("spec tree contains no LoopSpec")
    depths = {depth for _, depth in loops}
    if depths != {0}:
        raise TripCountAuditError(
            f"nested loops are not supported (levels present: {sorted(depths)})"
        )
    counts = {_concrete_count(loop.count) for loop, _ in loops}
    if len(counts) != 1:
        raise TripCountAuditError(f"loops disagree on trip count: {sorted(counts)}")
    count = counts.pop()
    if count < 1:
        raise TripCountAuditError(f"trip count must be positive, got {count}")
    return count


def _tiled_axis(arg: TensorArg, element_advance: int, base_count: int) -> int:
    """Which axis of ``arg.device_size`` carries the trip count.

    Follows ``superdsc.py:1277-1286``, which identifies the tiled axis from the
    per-iteration element advance: in a row-major device layout, one row of
    ``axis`` spans ``prod(device_size[axis + 1:])`` elements, so an advance of
    ``element_advance`` is ``rows_per_tile`` rows of that axis. (superdsc's own
    version of the test assumes ``rows_per_tile == 1``; here the tile may be
    several rows deep -- ``tile_size=2`` on a ``[1, 16, 64]`` buffer advances
    128 -- so divide it out rather than requiring equality.)

    The extent of the tiled axis is then ``rows_per_tile * count`` by
    construction, which is the invariant this actually keys on and the one that
    has to hold for the rewrite below to be the extent a real trace produces.

    Requires exactly one match. Zero or several means the count cannot be
    attributed to a single buffer axis, and the rewrite would be a guess.
    """
    sizes = [int(s) for s in arg.device_size]
    candidates = []
    for axis, extent in enumerate(sizes):
        inner = math.prod(sizes[axis + 1 :])
        if inner == 0 or element_advance % inner:
            continue
        rows_per_tile = element_advance // inner
        if rows_per_tile and extent == rows_per_tile * base_count:
            candidates.append(axis)
    if len(candidates) != 1:
        raise TripCountAuditError(
            f"cannot identify the tiled axis of arg {arg.arg_index} "
            f"({arg.name!r}): device_size={sizes}, element_advance="
            f"{element_advance}, base_count={base_count} matched "
            f"{len(candidates)} axes ({candidates})"
        )
    return candidates[0]


class PrefixIssue(NamedTuple):
    """A tiled input whose leading ``count`` rows are not a contiguous prefix."""

    arg_index: int
    name: str | None
    device_size: list[int]
    axis: int

    def __str__(self) -> str:
        return (
            f"arg {self.arg_index} ({self.name!r}): tiled axis {self.axis} of "
            f"device_size={self.device_size} sits behind non-unit extents "
            f"{self.device_size[: self.axis]}"
        )


def check_prefix_addressable(specs: list[Any]) -> list[PrefixIssue]:
    """Which tiled inputs cannot be fed an oversized buffer.

    Emitting the descriptor at another count is one claim; handing it a
    *max-size* buffer at launch is a second, independent one. The second holds
    only when the count axis is outermost in the buffer -- every axis before it a
    unit extent -- so that the descriptor's leading ``count`` rows are the same
    bytes as a standalone ``count``-row buffer. When an axis before it has
    extent > 1, its stride is count-derived and the oversized buffer's rows are
    further apart than the descriptor believes.

    An arg listed here must be passed at exactly the traced count, which for a
    shared trace means it cannot be a plain ``[count, ...]`` tensor. Caller's
    decision, so this reports rather than raises.

    Args:
        specs: A spec tree, with a concrete count or a symbolic one.

    Returns:
        One :class:`PrefixIssue` per offending ``(op, arg)``; empty when every
        tiled input is prefix-addressable.
    """
    issues: list[PrefixIssue] = []
    for loop, _ in _iter_loops(specs):
        base_count = _count_for_sizing(loop.count)
        for entry in loop.body:
            if not isinstance(entry, OpSpec) or not entry.tiled_symbol_trip_counts:
                continue
            syms = [sym for level in entry.tiled_symbols for sym in level]
            for arg in entry.args:
                if not isinstance(arg, TensorArg):
                    continue
                if arg.device_tile_advance_expr is None or not arg.is_input:
                    continue
                for sym in syms:
                    coeff = coeff_through_floor(arg.device_tile_advance_expr, sym)
                    if not coeff:
                        continue
                    sizes = [int(s) for s in arg.device_size]
                    axis = _tiled_axis(arg, int(coeff), base_count)
                    if any(sizes[a] != 1 for a in range(axis)):
                        issues.append(PrefixIssue(arg.arg_index, arg.name, sizes, axis))
    return issues
