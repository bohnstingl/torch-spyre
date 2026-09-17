# Copyright 2026 The Torch-Spyre Authors.
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

"""Does leaving a trip count symbolic change what the frontend produces?

The whole claim of ``_inductor/trip_count_symbol.py`` is that it does not: with
the count left to code generation, one trace serves every count, and the artifact
for count ``k`` is the max-count trace's artifact with the MLIR loop bound moved
and nothing else touched. Both halves of that are exact properties, not
tolerances, so both are pinned by byte comparison against a real trace:

* **The knob changes nothing.** Substituting the maximum back into a symbolic
  trace must reproduce a knob-off trace at that count **byte for byte** -- every
  descriptor, every address. This is the test that catches the symbol leaking
  into a buffer extent or an advance, which is the failure mode that would
  otherwise reach the device as wrong numerics with no error.
* **A smaller count moves the bound alone.** Substituting ``k < N`` must differ
  from that same trace in ``bundle.mlir`` only, and there only in the loop-bound
  constant. That is what makes it safe to hand one max-size buffer to every
  count; it can only ever run fewer iterations of the traced program. (The
  numerics of doing so are covered on device by ``test_for_each_tile.py``.)

Growing *past* the traced maximum is the one thing a symbolic trace still cannot
do, since the descriptors are sized for it -- so that is pinned as a refusal, and
a symbolic count reaching MLIR generation is pinned as a loud failure rather than
a guessed constant.

``TestAmbientTripCount`` then covers the channel that carries a count to a
compile and a launch (``execution/trip_count.py``), which is the other half a
caller touches. The numerics of an armed count are on device in
``test_for_each_tile_e2e.py``.

Needs a Spyre device to *trace* -- the operands are device tensors -- but no
backend compiler and no launch: emission goes through ``bundle_op_specs``, and
``capture_kernels(no_execute=True)`` stubs ``prepare_kernel``/``launch_jobplan``.
So, like every Spyre test, do not run it in parallel with another.
"""

import difflib
import filecmp
import os
import sys
import tempfile
import threading
import unittest

import sympy
import torch

import torch_spyre  # noqa: F401  -- registers the "spyre" device
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.op_spec import LoopSpec
from torch_spyre._inductor.trip_count_audit import (
    TripCountAuditError,
    check_prefix_addressable,
    trip_count_of,
)
from torch_spyre._inductor.trip_count_symbol import (
    substitute_trip_counts,
    symbolic_count_maxima,
    trip_count_symbol,
)
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.execution.trip_count import (
    TripCountUnsetError,
    apply_ambient_count,
    current_trip_count,
    precompile,
    trip_count,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "op_specs"))

from capture import capture_kernels  # noqa: E402
from runner import bundle_op_specs, pin_bundle_symbolic_args  # noqa: E402

MAX_COUNT = 8
TARGET_COUNT = 4


# ---------------------------------------------------------------------------
# The two shapes
# ---------------------------------------------------------------------------

TOY_K, TOY_N, TOY_TILE = 12, 6, 2


def trace_map_mode(count: int) -> None:
    """Map mode, ``out_dim=0``, a multi-row tile.

    ``tile_size=2`` makes the tiled axis advance two rows per iteration, which is
    the case superdsc's own one-row candidate test does not cover.
    """

    def fn(x, y):
        def body(_, ops):
            x_tile, y_whole = ops
            return None, x_tile @ y_whole

        _, out = for_each_tile(
            body, (x, y), dims=(0, None), tile_size=TOY_TILE, out_dim=0
        )
        return out

    x = torch.randn(count * TOY_TILE, TOY_K).half().to(DEVICE_NAME)
    y = torch.randn(TOY_K, TOY_N).half().to(DEVICE_NAME)
    torch.compile(fn, dynamic=False, fullgraph=True)(x, y)


RED_ROWS, RED_COLS = 8, 64


def trace_reduction_mode(count: int) -> None:
    """Paged attention's shape: reduction, a 3-leaf carry, ``tile_size=1``.

    Two dim-0-tiled operands and one untiled, an online-softmax-shaped carry, no
    stacked output -- and, as in paged attention, a tiled operand reached through
    ``tile.transpose(0, 1).reshape(...)``, which is what puts the count axis
    behind a non-unit extent in the device layout.
    """

    def fn(vals, mask, weights):
        def body(carry, tiles):
            run_max, run_sum, run_out = carry
            val_tile, mask_tile, w = tiles
            scores = val_tile.reshape(RED_ROWS, RED_COLS)
            scores = scores + mask_tile.transpose(0, 1).reshape(RED_ROWS, RED_COLS)
            tile_max = torch.amax(scores, dim=-1, keepdim=True)
            new_max = torch.maximum(run_max, tile_max)
            rescale = torch.exp(run_max - new_max)
            probs = torch.exp(scores - new_max)
            new_sum = run_sum * rescale + probs.sum(dim=-1, keepdim=True)
            new_out = run_out * rescale + torch.matmul(probs, w)
            return (new_max, new_sum, new_out), None

        state = {"dtype": vals.dtype, "device": vals.device}
        (_, run_sum, run_out), _ = for_each_tile(
            body,
            (vals, mask, weights),
            dims=(0, 0, None),
            tile_size=1,
            init=(
                torch.full((RED_ROWS, 1), float("-inf"), **state),
                torch.zeros((RED_ROWS, 1), **state),
                torch.zeros((RED_ROWS, RED_COLS), **state),
            ),
        )
        return run_out / run_sum

    vals = torch.randn(count, RED_ROWS, RED_COLS).half().to(DEVICE_NAME)
    mask = torch.zeros(count, RED_ROWS, RED_COLS).half().to(DEVICE_NAME)
    weights = torch.randn(RED_COLS, RED_COLS).half().to(DEVICE_NAME)
    torch.compile(fn, dynamic=False, fullgraph=True)(vals, mask, weights)


SHAPES = {"map_mode": trace_map_mode, "reduction_mode": trace_reduction_mode}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _capture_looped(trace, count: int, *, symbolic: bool) -> list:
    """Trace at ``count`` and return the records holding one loop.

    ``torch._dynamo.reset()`` first: without it the second trace in a process
    hits the code cache, compiles nothing, and the comparison silently becomes
    a trace against itself.

    A symbolic trace has to be made with a count armed even though it bakes
    none: ``sdsc`` substitutes for the artifact it compiles right there, and
    refuses to pick a count for itself (see ``apply_ambient_count``). Arming the
    maximum is deliberate -- it makes that first artifact the one every byte
    comparison below is against.
    """
    torch._dynamo.reset()
    with spyre_config.patch(spyre_trip_count_variants=symbolic):
        with capture_kernels(no_execute=True) as records:
            with trip_count(count):
                trace(count)
    return [r for r in records if any(isinstance(e, LoopSpec) for e in r.specs)]


def _emit(rec, specs, out_dir: str) -> None:
    # One name for every emission, so nothing name-derived can differ.
    pin_bundle_symbolic_args(rec.bundle_symbolic_args)
    bundle_op_specs("k", specs, out_dir, pool_size=rec.pool_size)


def _diff(a: str, b: str) -> list[str]:
    names = sorted(set(os.listdir(a)) | set(os.listdir(b)))
    out = []
    for name in names:
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        if not os.path.exists(pa):
            out.append(f"only in {b}: {name}")
        elif not os.path.exists(pb):
            out.append(f"only in {a}: {name}")
        elif not filecmp.cmp(pa, pb, shallow=False):
            out.append(name)
    return out


def _changed_lines(a: str, b: str, name: str) -> list[str]:
    """The added/removed lines of one file, for asserting *what* moved."""
    with open(os.path.join(a, name)) as fa, open(os.path.join(b, name)) as fb:
        hunk = difflib.unified_diff(fa.readlines(), fb.readlines(), n=0)
    return [
        line.rstrip()
        for line in hunk
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def _explain(a: str, b: str, names: list[str], limit: int = 24) -> str:
    """The first differing lines, so a failure names the field it is about.

    A bare list of filenames says a count-dependent field was missed but not
    which, and the answer is what the fix needs.
    """
    lines: list[str] = []
    for name in names:
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        with open(pa) as fa, open(pb) as fb:
            hunk = list(difflib.unified_diff(fa.readlines(), fb.readlines(), a, b))
        lines.append(f"--- {name} ---")
        lines.extend(line.rstrip() for line in hunk[:limit])
        if len(hunk) > limit:
            lines.append(f"... {len(hunk) - limit} more lines")
    return "\n".join(lines)


class TestSymbolicCountAgainstStaticTrace(unittest.TestCase):
    """A symbolic trace, substituted, against a real trace of the same program."""

    def _traces(self, shape: str):
        trace = SHAPES[shape]
        static = _capture_looped(trace, MAX_COUNT, symbolic=False)
        symbolic = _capture_looped(trace, MAX_COUNT, symbolic=True)
        self.assertTrue(static, f"{shape}: no LoopSpec-bearing kernel was traced")
        self.assertEqual(
            len(static),
            len(symbolic),
            f"{shape}: the two traces produced different kernel counts",
        )
        return static, symbolic

    def _check(self, shape: str) -> None:
        static, symbolic = self._traces(shape)
        with tempfile.TemporaryDirectory(prefix=f"tc_{shape}_") as work:
            for i, (rec_s, rec_y) in enumerate(zip(static, symbolic)):
                with self.subTest(kernel=i):
                    self.assertEqual(trip_count_of(rec_s.specs), MAX_COUNT)
                    maxima = symbolic_count_maxima(rec_y.specs)
                    self.assertEqual(
                        set(maxima.values()),
                        {MAX_COUNT},
                        "the knob was on, so this trace should have left its "
                        f"count to launch; it reports {maxima}",
                    )
                    self.assertEqual(
                        rec_s.pool_size,
                        rec_y.pool_size,
                        "the symbolic trace allocated a different pool, so the "
                        "count reached memory planning",
                    )

                    d = {
                        k: os.path.join(work, f"k{i}_{k}")
                        for k in ("static", "at_max", "at_target")
                    }
                    _emit(rec_s, rec_s.specs, d["static"])
                    _emit(
                        rec_y,
                        substitute_trip_counts(rec_y.specs, MAX_COUNT),
                        d["at_max"],
                    )
                    _emit(
                        rec_y,
                        substitute_trip_counts(rec_y.specs, TARGET_COUNT),
                        d["at_target"],
                    )

                    delta = _diff(d["static"], d["at_max"])
                    self.assertEqual(
                        delta,
                        [],
                        "leaving the count symbolic changed the artifacts at "
                        "the very count it was traced at, so the symbol reached "
                        "a descriptor\n" + _explain(d["static"], d["at_max"], delta),
                    )

                    delta = _diff(d["static"], d["at_target"])
                    self.assertEqual(
                        delta,
                        ["bundle.mlir"],
                        f"substituting {TARGET_COUNT} touched a descriptor; it "
                        "must move the loop bound and nothing else\n"
                        + _explain(d["static"], d["at_target"], delta),
                    )
                    # ...and within bundle.mlir, the bound line and no other.
                    # Without this the assertion above would accept any one-file
                    # difference, including a moved descriptor address.
                    changed = _changed_lines(d["static"], d["at_target"], "bundle.mlir")
                    normalized = [
                        line[0] + " " + " ".join(line[1:].split()) for line in changed
                    ]
                    self.assertEqual(
                        normalized,
                        [
                            f"- %loop_bound_0 = arith.constant {MAX_COUNT} : index",
                            f"+ %loop_bound_0 = arith.constant {TARGET_COUNT} : index",
                        ],
                        f"something other than the loop bound moved: {changed}",
                    )

    def test_map_mode(self):
        self._check("map_mode")

    def test_reduction_mode(self):
        self._check("reduction_mode")

    def test_a_symbolic_count_cannot_reach_mlir(self):
        """The backstop: unsubstituted, emission fails loudly.

        This is what makes the byte comparisons above trustworthy. A symbol that
        reached MLIR generation could only be turned into *some* constant, and a
        silently chosen count is the one outcome nothing downstream would catch.
        """
        _, symbolic = self._traces("map_mode")
        with tempfile.TemporaryDirectory(prefix="tc_backstop_") as work:
            with self.assertRaises(NotImplementedError):
                _emit(symbolic[0], symbolic[0].specs, os.path.join(work, "raw"))

    def test_prefix_addressability_is_reported(self):
        """A tiled operand behind a non-unit extent is reported, not raised on.

        Paged attention's mask is such an operand: its count axis sits behind the
        query axis, so its per-row base addresses are count-derived and it cannot
        be handed a max-size buffer at all. That is a launch-side fact for the
        caller to act on -- by padding it to the maximum -- rather than a failure
        of the substitution, so it is reported. The audit has to answer for a
        symbolic tree too, since that is now the only kind a variant-capable
        caller has.
        """
        static, symbolic = self._traces("reduction_mode")
        for rec in (*static, *symbolic):
            issues = check_prefix_addressable(rec.specs)
            self.assertTrue(
                all(any(s != 1 for s in i.device_size[: i.axis]) for i in issues),
                f"an issue was reported for a prefix-addressable arg: {issues}",
            )
        self.assertEqual(
            [str(i) for i in check_prefix_addressable(static[0].specs)],
            [str(i) for i in check_prefix_addressable(symbolic[0].specs)],
            "the audit's answer must not depend on whether the count is symbolic",
        )


class TestSubstitutionFailsClosed(unittest.TestCase):
    """Every count this cannot honour must raise, not guess."""

    def _tree(self, max_value: int = 8, loop_id: int = 0) -> list:
        return [LoopSpec(count=trip_count_symbol(loop_id, max_value), body=[])]

    def test_cannot_grow_past_the_traced_maximum(self):
        """Past the maximum every descriptor is too small -- the one hard limit."""
        with self.assertRaisesRegex(ValueError, "exceeds the maximum"):
            substitute_trip_counts(self._tree(4), 8)

    def test_any_count_up_to_the_maximum_is_allowed(self):
        """Including one *above* the count the first artifact was compiled for.

        The observable difference from respecializing a baked count, which could
        only ever shrink: there is no base count here to be below.
        """
        for k in (1, 2, 4, 8):
            with self.subTest(count=k):
                out = substitute_trip_counts(self._tree(8), k)
                self.assertEqual(out[0].count, k)

    def test_a_substituted_tree_reads_back_as_that_count(self):
        """The handoff: substitution yields ``sympy.Integer``, and every reader
        downstream of it -- the addressing audit, the MLIR emitter -- has to take
        that as a concrete count rather than as another expression."""
        out = substitute_trip_counts(self._tree(8), TARGET_COUNT)
        self.assertIsInstance(out[0].count, sympy.Integer)
        self.assertEqual(trip_count_of(out), TARGET_COUNT)

    def test_rejects_a_non_positive_or_non_int_count(self):
        for bad in (0, -1, True, 2.0, "4", None):
            with self.subTest(count=bad):
                with self.assertRaises(ValueError):
                    substitute_trip_counts(self._tree(), bad)

    def test_per_loop_counts_must_cover_every_loop(self):
        specs = self._tree(8, loop_id=0) + self._tree(8, loop_id=1)
        self.assertEqual(symbolic_count_maxima(specs), {0: 8, 1: 8})
        out = substitute_trip_counts(specs, {0: 2, 1: 4})
        self.assertEqual([s.count for s in out], [2, 4])
        with self.assertRaisesRegex(ValueError, "no trip count given for loop 1"):
            substitute_trip_counts(specs, {0: 2})

    def test_one_loop_with_two_maxima_is_refused(self):
        specs = self._tree(8, loop_id=0) + self._tree(4, loop_id=0)
        with self.assertRaisesRegex(ValueError, "two maxima"):
            symbolic_count_maxima(specs)

    def test_a_concrete_tree_is_left_alone(self):
        specs = [LoopSpec(count=8, body=[])]
        self.assertEqual(symbolic_count_maxima(specs), {})
        self.assertIs(substitute_trip_counts(specs, 4)[0], specs[0])

    def test_does_not_mutate_the_input(self):
        specs = self._tree(8)
        substitute_trip_counts(specs, 4)
        self.assertEqual(symbolic_count_maxima(specs), {0: 8})

    def test_trip_count_of_refuses_a_symbolic_tree(self):
        """It answers "what count was this built at"; a symbolic tree has none."""
        with self.assertRaisesRegex(TripCountAuditError, "symbolic"):
            trip_count_of(self._tree(8))

    def test_nested_loops(self):
        specs = [LoopSpec(count=4, body=[LoopSpec(count=4, body=[])])]
        with self.assertRaisesRegex(TripCountAuditError, "[Nn]ested"):
            trip_count_of(specs)

    def test_no_loop(self):
        with self.assertRaisesRegex(TripCountAuditError, "no LoopSpec"):
            trip_count_of([])


class TestApplyAmbientCount(unittest.TestCase):
    """The compile-time end of the ambient channel (``apply_ambient_count``)."""

    def test_a_concrete_tree_needs_no_count(self):
        specs = [LoopSpec(count=8, body=[])]
        out, count = apply_ambient_count(specs)
        self.assertIsNone(count)
        self.assertIs(out[0], specs[0])

    def test_substitutes_the_armed_count(self):
        specs = [LoopSpec(count=trip_count_symbol(0, 8), body=[])]
        with trip_count(2):
            out, count = apply_ambient_count(specs)
        self.assertEqual((out[0].count, count), (2, 2))

    def test_unarmed_compile_raises(self):
        """Compiling at the maximum instead would be silently wrong for anyone
        who armed nothing precisely because they wanted a smaller count."""
        specs = [LoopSpec(count=trip_count_symbol(0, 8), body=[])]
        with self.assertRaises(TripCountUnsetError):
            apply_ambient_count(specs)


class TestAmbientTripCount(unittest.TestCase):
    """The launch-time channel itself: arming, nesting, and refusing.

    Ambient state is a hazard, so what this pins down is not that arming works
    but that it cannot be left armed. Anything on device is covered by
    ``test_for_each_tile_e2e.py``; nothing here needs a kernel.
    """

    def test_unset_by_default(self):
        self.assertIsNone(current_trip_count())

    def test_arms_and_restores(self):
        with trip_count(4):
            self.assertEqual(current_trip_count(), 4)
        self.assertIsNone(current_trip_count())

    def test_nests(self):
        with trip_count(8):
            with trip_count(2):
                self.assertEqual(current_trip_count(), 2)
            self.assertEqual(current_trip_count(), 8)

    def test_restores_after_an_exception(self):
        """A raising kernel must not leave its count armed for the next launch."""
        with self.assertRaises(ZeroDivisionError):
            with trip_count(4):
                1 / 0
        self.assertIsNone(current_trip_count())

    def test_rejects_a_non_positive_or_non_int_count(self):
        for bad in (0, -1, True, 2.0, "4", None):
            with self.subTest(count=bad):
                with self.assertRaises(ValueError):
                    with trip_count(bad):
                        pass
        self.assertIsNone(current_trip_count())

    def test_is_thread_local(self):
        """One thread's count must not decide another's launch."""
        seen: list[int | None] = []
        with trip_count(8):
            thread = threading.Thread(target=lambda: seen.append(current_trip_count()))
            thread.start()
            thread.join()
        self.assertEqual(seen, [None])

    def test_precompile_with_no_counts_is_a_noop(self):
        self.assertEqual(precompile([]), [])


if __name__ == "__main__":
    unittest.main()
