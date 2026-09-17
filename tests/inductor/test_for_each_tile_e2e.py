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

"""End-to-end Spyre-device tests for WhileLoop -> OpSpec/LoopSpec lowering.

Compiles each fixture, runs it on the Spyre device, and compares against a
CPU reference -- for IR-level / mocked-IR unit tests of the lowering
machinery itself, see test_for_each_tile_lowering.py.

Minimum coverage per docs/superpowers/specs/2026-09-09-while-loop-lowering-design.md:
1. Single carry (this file: test_carry_mode_split_k) -- currently XFAIL on a
   read-copy/stick-layout gap; see that test's own docstring.
2 (carry + Kind.SLICE tile-advancing input): covered implicitly by
   test_carry_mode_split_k, whose X/Y operands are both Kind.SLICE.
3. Kind.GATHER: covered by test_gather_mode_paged_pages, which gathers one
   page per trip from inside the body the way paged attention does.
4. Multiple independent carries: covered by test_carry_mode_online_softmax
   (carry = (m, denom, acc), an online-softmax flash-attention inner loop).
Cases 5 and 6 (nested for_each_tile and the deliberate-decline case) are
follow-on work -- tracked as open items rather than duplicated here, since
each needs its own fixture beyond what's vendored so far.

test_map_mode_split_m (map mode: Kind.SLICE + Kind.INVARIANT operands, a
stacking carry, no user carry) passes end to end with verified numerics and
is the case that exercises the full splice -> DimHint synthesis ->
coarse-tile -> single scf.for pipeline.

TestTripCountVariants then reuses the online-softmax fixture for a second
question: with config.spyre_trip_count_variants on the count is left symbolic
through the frontend and chosen at code generation, so one trace serves several
trip counts -- does an armed count actually give that count's answer on device?
The artifacts are pinned byte for byte in test_trip_count_symbolic.py; the device
is the only oracle for what they compute. See the class docstring.
"""

import unittest
from unittest.mock import patch

import torch
from torch._inductor.exc import InductorError

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor import config as spyre_config
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre.execution.trip_count import (
    TripCountUnsetError,
    precompile,
    trip_count,
)

from tests.inductor.for_each_tile_fixtures import (
    SOFTMAX_TILE_SIZE,
    attention_inputs,
    matmul_inputs,
    online_softmax_fn,
    online_softmax_reference,
    paged_gather_fn,
    paged_gather_inputs,
    paged_gather_kv_fn,
    paged_gather_kv_inputs,
    paged_gather_kv_reference,
    paged_gather_reference,
    split_k_fn,
    split_m_fn,
)


class TestForEachTileE2E(unittest.TestCase):
    # Spyre's matmul runs in fp16, so the reference has to be an fp16-faithful
    # one: cast the operands first, then accumulate in fp32 on CPU. Comparing
    # against the fp32 product of fp32 operands would fail on rounding alone,
    # independently of anything this test is meant to check.
    #
    # Operands must also be cast to fp16 BEFORE the host->device transfer, not
    # after: `t.to(DEVICE_NAME).half()` (transfer fp32, cast on device)
    # currently produces garbage on this backend for reasons unrelated to
    # while_loop lowering -- a plain `torch.compile`d `a @ b` reproduces it
    # without any for_each_tile involved. `t.half().to(DEVICE_NAME)` is the
    # idiom the rest of the compiled-op suite uses (see
    # tests/inductor/test_inductor_matmul.py, whose inputs are constructed
    # `dtype=torch.float16` up front).
    #
    # rtol is the binding constraint here: operand/output magnitudes are
    # O(1)-O(10), so rtol * |expected| dominates atol (which only matters
    # near zero).
    ATOL = 0.1
    RTOL = 0.1

    @staticmethod
    def _operands():
        (X, Y), _ = matmul_inputs()
        ref = (X.half().float()) @ (Y.half().float())
        return X.half().to(DEVICE_NAME), Y.half().to(DEVICE_NAME), ref

    def test_map_mode_split_m(self):
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_m_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    @unittest.expectedFailure
    def test_carry_mode_split_k(self):
        """Carry mode: accumulate a split-K matmul across tiles.

        XFAIL on a gap in read-copy layout reconciliation, downstream of and
        distinct from everything this test's own lowering path needs -- the
        accumulator carry itself is wired correctly and WSR's own tiled-
        reduction accumulator (coarse_tile_fill/combine on the K level) picks
        the K accumulation up as intended. Tracked as issue #4460.

        The gap: ``for_each_tile``'s ``xs`` leaves for ``dims=(-1, 0)`` are
        3-D, transposed, ``movedim``-derived views of the operands
        (``[4, 3, 8]`` stride ``[3, 1, 12]`` for X, ``[4, 3, 6]`` stride
        ``[18, 6, 1]`` for Y). The K-advancing reads of those leaves route
        through ``coarse_tile.py``'s read-copy machinery, which builds tile
        buffers whose own layouts (e.g. ``[8, 6, 3]`` stride ``[0, 1, 6]`` --
        a broadcast leading dim over transposed inner dims) then fail stick
        reconciliation in ``optimize_restickify.py``/``propagate_layouts.py``
        ("No mechanism to scatter elements from one stick to multiple
        sticks"). The equivalent HINT-driven K-tiled matmul (same M/K/N,
        ``spyre_hint(num_tiles_per_dim={"K": 4})``) compiles and is
        numerically correct, and needs no read copies at all -- it reads the
        2-D operands directly. So this is a read-copy/stick-layout gap
        surfaced by the 3-D stacked-leaf shape, not a while_loop-lowering
        one, and it needs the same kind of layout work that the map-mode
        carry's own ``[4, 2, 6] -> [8, 6]`` fold needed (see
        ``while_loop_bridge.fold_stacked_carry_layout``) applied to the
        read side.
        """
        X_spyre, Y_spyre, ref = self._operands()

        compiled = torch.compile(split_k_fn, backend="inductor", fullgraph=True)
        out = compiled(X_spyre, Y_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_carry_mode_online_softmax(self):
        """Carry mode: 3-leaf carry (m, denom, acc), online-softmax over K/V tiles.

        Case 4 (multiple independent carries) from the design spec's minimum
        coverage list. carry_bindings_for/splice_while_loop's per-binding loop
        is already generic over an arbitrary-length carry list; this is the
        first fixture that actually drives a 3-leaf init= end to end, both to
        confirm the pytree carry survives decompose_scan_to_while_loop's
        scan -> while_loop decomposition intact, and to confirm
        _extra_readers_of_placeholder/_snapshot_carry_placeholder correctly
        handle the write-after-read hazard this body's own m carry hits:
        `correction = exp(m - m_new)` reads m's OLD value a second time,
        after m_new (m's per-iteration output) has already been computed --
        the exact case an in-place-only rewrite would silently corrupt.
        """
        Q, K, V = attention_inputs()
        ref = online_softmax_reference(Q, K, V)

        Q_spyre = Q.to(DEVICE_NAME)
        K_spyre = K.to(DEVICE_NAME)
        V_spyre = V.to(DEVICE_NAME)

        compiled = torch.compile(online_softmax_fn, backend="inductor", fullgraph=True)
        out = compiled(Q_spyre, K_spyre, V_spyre)

        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_gather_mode_paged_pages(self):
        """Kind.GATHER: the body gathers its own page, one per trip.

        Case 3 from the design spec's minimum coverage list, and the shape
        paged attention actually wants: tile the block table, keep the page
        pool invariant, and let the body read its page index out of the tile.
        The index is a point read whose address advances with the spliced loop
        and has no iteration dim, so this drives
        coarse_tile._point_splice_advance_for_dep (record the per-trip
        advance), _full_buffer_read_deps' point-read exclusion (leave the read
        direct instead of staging a 1-element int32 into scratch),
        _rebase_point_splice_reads (pin the index to iteration 0 so the
        advance is not applied twice), and insert_restickify's per-dep advance
        handover. Numerics, not just compilation: every one of those can be
        got wrong in a way that compiles and re-reads the same page.

        Two matmuls per trip and a distinct page per trip, so an advance
        applied to the wrong operand or dropped entirely shows up as a large
        mismatch rather than rounding.
        """
        pages, table, q = paged_gather_inputs()
        ref = paged_gather_reference(pages, q)

        compiled = torch.compile(paged_gather_fn, backend="inductor", fullgraph=True)
        out = compiled(pages.to(DEVICE_NAME), table.to(DEVICE_NAME), q.to(DEVICE_NAME))

        # Looser than the class defaults: the accumulator sums four
        # score-weighted pages of magnitude ~sqrt(head_size), so fp16 matmul
        # rounding alone reaches a couple of absolute units here.
        torch.testing.assert_close(out.cpu().float(), ref, atol=2.0, rtol=0.05)

    def test_gather_mode_paged_pages_kv(self):
        """Kind.GATHER with separate K and V pools: one marker, two consumers.

        Numerics for the shape spyre-inference's page_attn_kernel actually
        has (see paged_gather_kv_fn): the page index sliced out of the tiled
        block table feeds an index_select per cache, so the table's single
        tile_dim_marker has two consuming reads. Compilation alone is
        covered device-lessly by
        TestConsumeTileDimMarkers.test_marker_with_two_computed_buffer_
        consumers_maps_both; what only the device can show is whether the
        marker's per-trip advance was composed into BOTH gathers. Getting
        that wrong for one of them re-reads page PAGE_ORDER[0]'s K (or V)
        on every trip -- a large mismatch here, and invisible on CPU.
        """
        k_pages, v_pages, table, q = paged_gather_kv_inputs()
        ref = paged_gather_kv_reference(k_pages, v_pages, q)

        compiled = torch.compile(paged_gather_kv_fn, backend="inductor", fullgraph=True)
        out = compiled(
            k_pages.to(DEVICE_NAME),
            v_pages.to(DEVICE_NAME),
            table.to(DEVICE_NAME),
            q.to(DEVICE_NAME),
        )

        # Same tolerance rationale as test_gather_mode_paged_pages above.
        torch.testing.assert_close(out.cpu().float(), ref, atol=2.0, rtol=0.05)


class TestTripCountVariants(unittest.TestCase):
    """One trace, several trip counts, chosen ambiently at launch.

    With ``config.spyre_trip_count_variants`` on, the frontend leaves the count
    symbolic (``_inductor/trip_count_symbol.py``) and each count is produced by
    substituting it into the retained spec tree and re-emitting, instead of by
    tracing again. The count then reaches the kernel through a thread-local the
    caller arms, because it can travel neither in the call arguments nor in the
    graph -- see ``execution/trip_count.py``.

    Every count addresses the *maximum*'s buffers and only the loop bound moves,
    so the caller hands the same max-size operands to all of them.
    ``test_trip_count_symbolic.py`` pins that byte for byte on the artifacts,
    which says nothing about what the device then does with them -- and that is
    what these tests are for: running count ``k`` over max-size buffers has to
    give exactly the answer for the first ``k`` tiles.
    """

    MAX_COUNT = 8
    # Ascending, so the count the first artifact happens to be compiled for is
    # the *smallest* one: with the count symbolic there is no base count to
    # shrink from, and every later launch in the ladder grows past it.
    COUNTS = (1, 2, 4, 8)
    ATOL, RTOL = 0.1, 0.1

    def _operands(self):
        Q, K, V = attention_inputs(self.MAX_COUNT)
        return (
            (Q, K, V),
            (Q.to(DEVICE_NAME), K.to(DEVICE_NAME), V.to(DEVICE_NAME)),
        )

    @staticmethod
    def _reference(host, count):
        """The answer for ``count`` tiles: the reference on that KV prefix."""
        Q, K, V = host
        end = count * SOFTMAX_TILE_SIZE
        return online_softmax_reference(Q, K[:end], V[:end])

    @staticmethod
    def _compile():
        # Without the reset the second compile in a process hits Dynamo's code
        # cache, so a knob flipped between compiles would not be read.
        torch._dynamo.reset()
        return torch.compile(online_softmax_fn, backend="inductor", fullgraph=True)

    def test_ambient_count_selects_the_variant(self):
        """Each armed count must give that count's answer, on shared buffers.

        Two passes over the ladder, not one: the first builds each variant, the
        second relaunches an already-prepared jobplan. A descriptor whose
        per-iteration advance is not rewound between launches, or a carry left
        holding the previous launch's state, is correct on the first launch and
        wrong on the second -- so a single pass would not see it.
        """
        host, dev = self._operands()
        refs = {k: self._reference(host, k) for k in self.COUNTS}

        # Non-vacuity: the counts must have visibly different answers, or
        # selecting the wrong variant would pass unnoticed.
        spread = (refs[1] - refs[self.MAX_COUNT]).abs().max().item()
        self.assertGreater(
            spread,
            10 * self.ATOL,
            "the trip counts' answers are too close for this test to detect a "
            "variant selected wrongly",
        )

        with spyre_config.patch(spyre_trip_count_variants=True):
            compiled = self._compile()
            for pass_no in (1, 2):
                for count in self.COUNTS:
                    with self.subTest(pass_no=pass_no, count=count):
                        with trip_count(count):
                            out = compiled(*dev)
                        torch.testing.assert_close(
                            out.cpu().float(),
                            refs[count],
                            atol=self.ATOL,
                            rtol=self.RTOL,
                        )

    def test_a_count_larger_than_the_compiled_one(self):
        """Growing past the first artifact's count is the point of the symbol.

        Respecializing a *concrete* tree could only ever shrink, because its
        descriptors were sized for the count it was traced at. A symbolic tree
        was sized for the maximum and compiled for whichever count happened to
        be armed first, so growth is ordinary -- and that is the observable
        proof that the frontend really is count-agnostic rather than merely
        re-emitting a count it had already seen.
        """
        host, dev = self._operands()
        with spyre_config.patch(spyre_trip_count_variants=True):
            compiled = self._compile()
            with trip_count(2):
                compiled(*dev)
            with trip_count(self.MAX_COUNT):
                out = compiled(*dev)
        torch.testing.assert_close(
            out.cpu().float(),
            self._reference(host, self.MAX_COUNT),
            atol=self.ATOL,
            rtol=self.RTOL,
        )

    def test_the_frontend_runs_once_for_the_whole_ladder(self):
        """The saving itself: N counts cost one frontend, not N.

        Counted at ``SpyreAsyncCompile.sdsc`` -- the frontend's last step, and
        the one call a variant does *not* go through, since ``build_variant``
        re-emits from the retained tree straight into ``generate_bundle``. So a
        count that re-traced would show up here as a second set of calls.
        """
        _, dev = self._operands()
        calls: list[str] = []
        real_sdsc = SpyreAsyncCompile.sdsc

        def counting_sdsc(self, kernel_name, specs, pool_size=0):
            calls.append(kernel_name)
            return real_sdsc(self, kernel_name, specs, pool_size=pool_size)

        with spyre_config.patch(spyre_trip_count_variants=True):
            with patch.object(SpyreAsyncCompile, "sdsc", counting_sdsc):
                compiled = self._compile()
                with trip_count(self.COUNTS[0]):
                    compiled(*dev)
                after_first_launch = len(calls)
                for count in self.COUNTS[1:]:
                    with trip_count(count):
                        compiled(*dev)

        self.assertTrue(after_first_launch, "no kernel was compiled at all")
        self.assertEqual(
            len(calls),
            after_first_launch,
            f"the frontend ran again for a later count: {calls[after_first_launch:]}",
        )

    def test_unarmed_compile_raises(self):
        """With nothing armed there is no count to compile *for*, so refuse.

        Falling back to the maximum would hand back an artifact that runs every
        iteration -- indistinguishable from a correct one until its output is
        wrong -- to the one caller least likely to want it.

        The refusal arrives wrapped, unlike the launch-side one below: it happens
        while Inductor is executing the generated wrapper, so Inductor re-raises
        it as its own ``InductorError``.
        """
        _, dev = self._operands()
        with spyre_config.patch(spyre_trip_count_variants=True):
            compiled = self._compile()
            with self.assertRaises(InductorError) as caught:
                compiled(*dev)
        self.assertIsInstance(caught.exception.inner_exception, TripCountUnsetError)
        self.assertIn("no trip count armed", str(caught.exception))

    def test_unarmed_launch_raises(self):
        """An already-compiled kernel must refuse an unarmed launch too.

        A caller that passes max-size buffers and forgets the ``with`` would
        otherwise silently run some earlier count over rows it never wrote,
        which is a wrong answer with no error -- so the default is refusal, at
        launch as much as at compile time.
        """
        _, dev = self._operands()
        with spyre_config.patch(spyre_trip_count_variants=True):
            compiled = self._compile()
            with trip_count(2):
                compiled(*dev)
            with self.assertRaisesRegex(TripCountUnsetError, "no trip count"):
                compiled(*dev)

    def test_knob_off_ignores_the_ambient_count(self):
        """Off, the count is inert: the traced kernel runs, armed or not.

        Structural rather than a runtime branch: a knob-off trace bakes its
        count, so the tree it retains has no symbolic count and the runner is
        never variant-capable in the first place. That is what makes a stray
        ``with trip_count(...)`` left in a host's dispatch path harmless.
        """
        host, dev = self._operands()
        ref = self._reference(host, self.MAX_COUNT)
        with spyre_config.patch(spyre_trip_count_variants=False):
            compiled = self._compile()
            with trip_count(1):
                out = compiled(*dev)
        torch.testing.assert_close(
            out.cpu().float(), ref, atol=self.ATOL, rtol=self.RTOL
        )

    def test_precompile_reports_every_count(self):
        """Warmup builds the ladder ahead of use, and reports what it did.

        Failures come back as reports rather than exceptions on purpose -- an
        unbuilt variant is compiled on first use, so warmup must not take down a
        host that would otherwise run correctly, just slower.
        """
        _, dev = self._operands()
        with spyre_config.patch(spyre_trip_count_variants=True):
            compiled = self._compile()
            # One launch first: the runner registers itself at construction,
            # which happens when the generated wrapper is loaded.
            with trip_count(self.MAX_COUNT):
                compiled(*dev)

            reports = precompile(self.COUNTS)
            self.assertTrue(reports, "no variant-capable kernel was registered")
            self.assertEqual(
                [r for r in reports if r.status == "failed"],
                [],
                "\n".join(str(r) for r in reports),
            )
            for count in self.COUNTS:
                self.assertIn(
                    count,
                    {r.count for r in reports},
                    f"warmup skipped count {count}",
                )


if __name__ == "__main__":
    unittest.main()
