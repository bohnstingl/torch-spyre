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

"""A loop trip count that stays symbolic through the frontend.

A ``for_each_tile`` loop's trip count is baked at trace time, so one count is one
full ``torch.compile``. The frontend does not actually depend on it: the artifacts
for two counts differ in the MLIR loop bound and nothing else, because the tiled
dim's per-iteration extent -- not the count -- is what sizes and addresses every
descriptor. Making the count a symbol lets one trace serve every count, with the
number supplied at code generation.

The design is one invariant:

    A trip count is symbolic in exactly one place -- ``LoopSpec.count``. Every
    other consumer sees the literal **maximum**. Code generation substitutes the
    concrete count.

That invariant is held *structurally* rather than by policing call sites.
``CoarseTileInfo.loop_count`` stays literal all the way through the ``wsr/``
passes, so the ~20 places that build advance extents and buffer sizes out of it
(``running * loop_count[level_idx]`` and friends) keep working untouched and
cannot produce a symbolic shape. What a spliced group carries instead is one
boolean, ``CoarseTileInfo.symbolic_trip_count``; ``scheduler.py``'s
``_loop_count`` -- the single reader that feeds ``LoopSpec.count`` -- mints the
symbol there, from the loop's group id and its literal count. Since minting is a
pure function of that pair, every node in a loop group mints the *same* symbol,
which is what ``scheduler.py``'s ``assert next_count == count`` needs.

So the frontend behaves exactly as a trace at the maximum count would, and the
only thing a smaller count changes is when the loop stops. That is the whole
correctness argument, and it is the same one the descriptor-preserving
respecialization this replaces already carried and had verified on device:
descriptors sized and addressed for ``N`` tiles, with the loop stopping after
``k <= N`` of them, can only ever run *fewer iterations of the traced program*.

What that argument does *not* cover is the buffers a caller passes. They must be
the maximum's, not the armed count's, and for one shape of operand an oversized
buffer is not even addressable -- see ``trip_count_audit.py``'s
``check_prefix_addressable``, which reports those.

Deliberately, no *shape* becomes symbolic. Only the loop bound does. Making the
tiled tensor dimension symbolic instead runs into a catalogue of unrelated
blockers (``s//(s//2)`` index forms in ``wsr/coarse_tile.py``'s read-copy
planning, unsortable symbolic strides in ``wsr/tile.py``); none of them are on
this path.

Why the maximum is encoded in the symbol's *name*: sympy interns symbols
globally by ``(class, name, assumptions)``, and that cache outlives a single
compile. Two kernels in one process can each have a loop 0 with different
maxima, so keying only on the loop id would either collide or force a spurious
conflict error. Encoding the maximum makes distinct maxima distinct symbols,
keeps ``(loop_id, max)`` identity-stable, and lets the maximum travel with the
symbol so it survives pickling and ``copy.deepcopy`` even though
``sympy.Symbol`` reconstructs subclass instances through ``__new__``.
"""

from collections.abc import Mapping
from typing import Any

import sympy

from .op_spec import LoopSpec


class TripCountSymbol(sympy.Symbol):
    """A loop trip count symbol carrying the maximum it was traced at.

    Construct through :func:`trip_count_symbol` rather than directly, so the
    name encoding stays in one place.
    """

    __slots__ = ("loop_id", "max_value")

    def __new__(cls, loop_id: int, max_value: int) -> "TripCountSymbol":
        if isinstance(loop_id, bool) or not isinstance(loop_id, int) or loop_id < 0:
            raise ValueError(f"loop_id must be a non-negative int, got {loop_id!r}")
        if isinstance(max_value, bool) or not isinstance(max_value, int):
            raise ValueError(f"max_value must be an int, got {max_value!r}")
        if max_value < 1:
            raise ValueError(f"max_value must be positive, got {max_value}")
        # integer/positive let sympy simplify arithmetic built on the count
        # (e.g. the advance-extent products) instead of leaving it unevaluated.
        obj = super().__new__(
            cls, f"_trip_{loop_id}_{max_value}", integer=True, positive=True
        )
        obj.loop_id = loop_id
        obj.max_value = max_value
        return obj

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # sympy's Symbol returns (name,) here, which would drop both attributes
        # and then fail our __new__ signature. Reconstruct from the pair instead.
        return ((self.loop_id, self.max_value), {})


def trip_count_symbol(loop_id: int, max_value: int) -> TripCountSymbol:
    """Return the trip-count symbol for loop ``loop_id`` traced at ``max_value``.

    Repeated calls with the same pair return the same object: sympy's symbol
    cache does the memoizing, keyed on the encoded name. The generated wrapper
    calls this by name to rebuild a count from source, so its signature is part
    of that on-disk format -- see ``spyre_kernel.py``'s ``_codegen_op_spec_list``.
    """
    return TripCountSymbol(loop_id, max_value)


def is_symbolic_count(expr: Any) -> bool:
    """Whether ``expr`` is a trip count still awaiting a concrete value."""
    return isinstance(expr, TripCountSymbol)


def _resolve(count: Any, counts: int | Mapping[int, int]) -> sympy.Integer:
    """The concrete count for one symbolic ``LoopSpec.count``."""
    loop_id = count.loop_id
    if isinstance(counts, Mapping):
        if loop_id not in counts:
            raise ValueError(
                f"no trip count given for loop {loop_id} (have {sorted(counts)})"
            )
        value = counts[loop_id]
    else:
        value = counts
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"trip count must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"trip count must be positive, got {value}")
    if value > count.max_value:
        # Growing past the traced maximum would read past the end of every
        # descriptor, which are all sized for the maximum.
        raise ValueError(
            f"trip count {value} exceeds the maximum {count.max_value} that "
            f"loop {loop_id} was traced at"
        )
    return sympy.Integer(value)


def _substitute_body(body: list[Any], counts: int | Mapping[int, int]) -> list[Any]:
    out: list[Any] = []
    changed = False
    for entry in body:
        if isinstance(entry, LoopSpec):
            new_entry = _substitute_loop(entry, counts)
            changed = changed or new_entry is not entry
            out.append(new_entry)
        else:
            out.append(entry)
    return out if changed else body


def _substitute_loop(loop: LoopSpec, counts: int | Mapping[int, int]) -> LoopSpec:
    body = _substitute_body(loop.body, counts)
    if not is_symbolic_count(loop.count):
        return loop if body is loop.body else LoopSpec(count=loop.count, body=body)
    return LoopSpec(count=_resolve(loop.count, counts), body=body)


def substitute_trip_counts(
    specs: list[Any], counts: int | Mapping[int, int]
) -> list[Any]:
    """Return ``specs`` with every symbolic ``LoopSpec.count`` made concrete.

    Only ``LoopSpec.count`` is touched. Every other field -- including
    ``OpSpec.tiled_symbol_trip_counts``, which the invariant keeps at the
    maximum -- is shared with the input rather than copied, so the emitted SDSC
    comes out identical to the traced maximum's and only the MLIR loop bound
    moves. That is the whole substitution.

    Args:
        specs: The retained spec tree, as handed to ``SpyreAsyncCompile.sdsc``.
        counts: One count for every loop, or a per-loop ``{loop_id: count}``.

    Returns:
        A new list; ``specs`` is not mutated. Op bodies are shared, and a subtree
        with no symbolic count is returned as-is.

    Raises:
        ValueError: A count is missing, not a positive int, or larger than the
            maximum its loop was traced at.
    """
    return _substitute_body(specs, counts)


def symbolic_count_maxima(specs: list[Any]) -> dict[int, int]:
    """``{loop_id: max_value}`` for every symbolic count in ``specs``.

    Lets a caller discover what it must supply -- and, when empty, that this
    tree was traced with concrete counts and needs no substitution.
    """
    found: dict[int, int] = {}

    def walk(body: list[Any]) -> None:
        for entry in body:
            if not isinstance(entry, LoopSpec):
                continue
            if is_symbolic_count(entry.count):
                loop_id, max_value = entry.count.loop_id, entry.count.max_value
                previous = found.get(loop_id)
                if previous is not None and previous != max_value:
                    raise ValueError(
                        f"loop {loop_id} appears with two maxima: "
                        f"{previous} and {max_value}"
                    )
                found[loop_id] = max_value
            walk(entry.body)

    walk(specs)
    return found
