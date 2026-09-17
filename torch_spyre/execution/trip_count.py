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

"""Select a ``for_each_tile`` kernel's trip count at launch, not at trace time.

A trip count reaches the device as a constant in ``bundle.mlir``, so today one
count is one full ``torch.compile``. This module makes it a *launch-time* choice
instead: the kernel is traced once with the count left symbolic (see
``_inductor/trip_count_symbol.py``), and each additional count is produced by
substituting it into the retained spec tree and re-emitting, which skips the
frontend -- 72-77% of a compile.

Because the retained tree never held a count in the first place, there is no
"base" count a variant has to be derived *from*: every count, including the one
the first artifact was compiled for, is the same substitution into the same tree.
The only ceiling is the maximum the trace was made at, which sized every
descriptor.

The count cannot travel in the call arguments, because the generated wrapper's
``run()`` signature and ``TensorArg.arg_index`` baking are fixed at trace time
and a count is neither an address nor a dimension (``SymbolicArg`` has no kind
for it). So it travels *ambiently*: the caller arms it around the call,

    with trip_count(k):
        compiled_model(...)

and every variant-capable kernel launched inside picks it up.

Ambient state is a hazard, so the two ways it can be wrong both fail loudly:

* **unset** -- launching a variant-capable kernel with no count armed raises,
  rather than falling back to the traced count. A caller that passes max-size
  buffers and forgets the ``with`` would otherwise silently run the full traced
  count over rows it never wrote.
* **unknown** -- a count with no compiled variant is compiled on the spot rather
  than rounded to one that exists. :func:`precompile` exists to move that cost
  into warmup, but skipping it only costs latency, never correctness.

Consequently, turning ``spyre_trip_count_variants`` on requires *every* looped
kernel's call site to arm a count, not just the one being optimized.
"""

import contextlib
import os
import threading
import time
import weakref
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import torch

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.trip_count_symbol import (
    substitute_trip_counts,
    symbolic_count_maxima,
)

from .kernel_cache import (
    _move_to_failed_dir,
    allocate_compile_dir,
    commit_compile_dir,
    compute_specs_hash,
    get_cached_kernel_dir,
    get_kernel_registry,
)

logger = get_inductor_logger("trip_count")

__all__ = [
    "TripCountUnsetError",
    "VariantReport",
    "current_trip_count",
    "precompile",
    "trip_count",
]


class TripCountUnsetError(RuntimeError):
    """A variant-capable kernel was launched with no ambient trip count."""


# ---------------------------------------------------------------------------
# The ambient count
# ---------------------------------------------------------------------------

# Thread-local rather than a global: a host may dispatch from more than one
# thread, and a global would let one thread's count decide another's launch.
_state = threading.local()


def current_trip_count() -> int | None:
    """The trip count armed on this thread, or ``None`` if none is."""
    return getattr(_state, "count", None)


@contextlib.contextmanager
def trip_count(count: int):
    """Arm ``count`` as the trip count for kernels launched in this block.

    Re-entrant, and restores the previous value in a ``finally`` so a raising
    kernel cannot leave a stale count armed for the next, unrelated launch.

    Args:
        count: A positive trip count.

    Raises:
        ValueError: ``count`` is not a positive int.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"trip count must be an int, got {count!r}")
    if count < 1:
        raise ValueError(f"trip count must be positive, got {count}")
    previous = getattr(_state, "count", None)
    _state.count = count
    try:
        yield
    finally:
        _state.count = previous


# ---------------------------------------------------------------------------
# Variant production
# ---------------------------------------------------------------------------


class _Pending(NamedTuple):
    """A variant whose bundle is written but whose backend compile has not run."""

    kernel_name: str
    count: int
    compile_dir: str
    cache_key: str | None


def _use_cache() -> bool:
    return (
        _spyre_config.spyre_kernel_cache
        and not torch._inductor.config.force_disable_caches
    )


def variant_kernel_name(kernel_name: str, count: int) -> str:
    return f"{kernel_name}_tc{count}"


def prepare_variant(
    kernel_name: str,
    specs: Sequence[Any],
    count: int,
    pool_size: int,
) -> str | _Pending:
    """Emit the bundle for ``count``, or return its cached directory.

    Everything up to and including bundle emission, which is the cheap half:
    ``generate_bundle``'s self-time is 0.03 s against DeepTools' seconds. Split
    from :func:`finish_variant` so a batch can emit serially and then compile in
    parallel.

    Returns:
        A ready ``code_dir`` on a cache hit, or a :class:`_Pending` to hand to
        :func:`finish_variant`.

    Raises:
        ValueError: ``count`` is not positive, or exceeds the maximum the tree
            was traced at.
    """
    from torch_spyre._inductor.codegen.bundle import generate_bundle

    variant_specs = substitute_trip_counts(list(specs), count)
    name = variant_kernel_name(kernel_name, count)

    cache_key = None
    if _use_cache():
        try:
            cache_key = compute_specs_hash(
                variant_specs, kernel_name=name, pool_size=pool_size
            )
        except RuntimeError as exc:
            logger.warning(
                "Kernel cache disabled for %s: could not compute cache key: %s",
                name,
                exc,
            )
        else:
            cached_dir = get_cached_kernel_dir(cache_key)
            if cached_dir is not None:
                get_kernel_registry().record_hit(cache_key)
                logger.debug("Variant cache HIT: %s -> %s", name, cached_dir)
                return cached_dir
            get_kernel_registry().record_miss(cache_key)

    if cache_key is None:
        from .async_compile import get_output_dir

        compile_dir = get_output_dir(name)
    else:
        compile_dir = allocate_compile_dir(cache_key)

    try:
        generate_bundle(name, compile_dir, variant_specs, pool_size=pool_size)
    except Exception:
        if cache_key is not None:
            _move_to_failed_dir(compile_dir)
        raise
    return _Pending(name, count, compile_dir, cache_key)


def finish_variant(pending: _Pending) -> str:
    """Run the backend compiler on an emitted bundle and return its ``code_dir``."""
    from .async_compile import _run_dxp

    try:
        _run_dxp(pending.kernel_name, pending.compile_dir, dict(os.environ))
    except Exception:
        if pending.cache_key is not None:
            _move_to_failed_dir(pending.compile_dir)
        raise
    if pending.cache_key is None:
        return pending.compile_dir
    return commit_compile_dir(pending.compile_dir, pending.cache_key)


def build_variant(
    kernel_name: str,
    specs: Sequence[Any],
    count: int,
    pool_size: int,
) -> str:
    """Produce the ``code_dir`` for ``count``, compiling it if needed."""
    prepared = prepare_variant(kernel_name, specs, count, pool_size)
    if isinstance(prepared, str):
        return prepared
    return finish_variant(prepared)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

# Variant-capable runners, so warmup can reach them without the host having to
# hold references. Weak, so a runner dropped with its compiled graph does not
# keep its spec tree alive.
_runners: "weakref.WeakSet[Any]" = weakref.WeakSet()
_runners_lock = threading.Lock()


def register_runner(runner: Any) -> None:
    with _runners_lock:
        _runners.add(runner)


class VariantReport(NamedTuple):
    """The outcome of one ``(kernel, count)`` warmup build."""

    kernel_name: str
    count: int
    status: str  # "built" | "cached" | "failed"
    detail: str = ""

    def __str__(self) -> str:
        tail = f": {self.detail}" if self.detail else ""
        return f"{self.kernel_name} count={self.count} {self.status}{tail}"


def precompile(
    counts: Iterable[int], *, kernel_name_prefix: str | None = None
) -> list[VariantReport]:
    """Build every variant in ``counts`` ahead of the first launch that needs it.

    Purely an optimization: a count left out is compiled on first use. Bundles
    are emitted serially in this process and then handed to the backend compiler
    in parallel, since that is the expensive half and it is a subprocess.

    Args:
        counts: The trip counts to build.
        kernel_name_prefix: Restrict to runners whose kernel name starts with
            this, for warming one kernel out of a graph that has several.

    Returns:
        One :class:`VariantReport` per ``(kernel, count)`` attempted. A failed
        build is reported, not raised: warmup should not take down a host that
        would otherwise run correctly, just slower.
    """
    wanted = sorted({int(c) for c in counts})
    with _runners_lock:
        runners = list(_runners)

    reports: list[VariantReport] = []
    pending: list[tuple[Any, _Pending]] = []
    t_emit = time.time()

    for runner in runners:
        if kernel_name_prefix and not runner.kernel_name.startswith(kernel_name_prefix):
            continue
        for count in wanted:
            if runner.has_variant(count):
                reports.append(VariantReport(runner.kernel_name, count, "cached"))
                continue
            try:
                prepared = runner.prepare_variant(count)
            except Exception as exc:  # noqa: BLE001
                reports.append(
                    VariantReport(
                        runner.kernel_name,
                        count,
                        "failed",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if isinstance(prepared, str):
                runner.adopt_variant(count, prepared)
                reports.append(VariantReport(runner.kernel_name, count, "cached"))
            else:
                pending.append((runner, prepared))

    if not pending:
        return reports

    emit_s = time.time() - t_emit
    t_backend = time.time()
    workers = min(len(pending), os.cpu_count() or 1)
    if workers > 1 and _spyre_config.async_dxp_compile:
        # dxp_standalone is a subprocess, so threads overlap it fine. Emission
        # above stays serial: generate_bundle is not known to be thread-safe,
        # and at 0.03 s it is not where the time is.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = [(r, p, pool.submit(finish_variant, p)) for r, p in pending]
    else:
        outcomes = [(r, p, None) for r, p in pending]

    for runner, prep, future in outcomes:
        try:
            code_dir = future.result() if future is not None else finish_variant(prep)
        except Exception as exc:  # noqa: BLE001
            reports.append(
                VariantReport(
                    runner.kernel_name,
                    prep.count,
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        runner.adopt_variant(prep.count, code_dir)
        reports.append(VariantReport(runner.kernel_name, prep.count, "built"))

    # Logged once per batch that actually built something, at INFO: this feature
    # exists to buy compile time, so the split between the half it skips
    # (emission) and the half it still pays (the backend compiler) is the number
    # that says whether it is working.
    logger.info(
        "Built %d trip-count variants: emit %.2fs, backend %.2fs (%s).",
        len(pending),
        emit_s,
        time.time() - t_backend,
        f"{workers} workers"
        if workers > 1 and _spyre_config.async_dxp_compile
        else "serial",
    )
    return reports


def variant_capable(specs: Sequence[Any] | None) -> int | None:
    """The largest trip count ``specs`` can be launched at, or ``None``.

    The gate for a runner becoming variant-capable: specs were retained, and the
    tree actually left a count for launch to choose.

    It deliberately does *not* consult ``spyre_trip_count_variants``. The knob
    decides whether a *trace* leaves the count symbolic, and by the time a tree
    exists that question is settled: a tree with no symbolic count is not
    variant-capable no matter how the knob stands, and one that has a symbolic
    count needs an armed count to launch at all. Re-checking the knob here would
    get the awkward case exactly backwards -- a wrapper cached under the knob
    reloaded in a process without it still carries the symbol, and would then
    launch whichever single count it happened to be compiled for while silently
    ignoring the armed one.

    A kernel with several spliced loops gets one symbol each, but the ambient
    channel carries a single count for all of them, so the ceiling is the
    *smallest* of their maxima -- past that, one loop's descriptors would be too
    small.
    """
    if not specs:
        return None
    try:
        maxima = symbolic_count_maxima(list(specs))
    except ValueError as exc:
        logger.debug("Not variant-capable: %s", exc)
        return None
    if not maxima:
        return None
    return min(maxima.values())


def apply_ambient_count(specs: Sequence[Any]) -> tuple[list[Any], int | None]:
    """Make ``specs`` concrete at the count armed on this thread.

    Code generation is where a symbolic count has to become a number, and
    ``SpyreAsyncCompile.sdsc`` is the one place both emitters receive the tree --
    so this is the substitution point, and the artifact it goes on to compile
    belongs to the returned count rather than to the trace.

    Returns:
        ``(specs, None)`` unchanged when the tree has no symbolic count -- the
        knob is off, or this kernel has no spliced loop. Otherwise the
        substituted tree and the count it was substituted at.

    Raises:
        TripCountUnsetError: The tree needs a count and none is armed. Compiling
            at the traced maximum instead would hand back an artifact that runs
            every iteration, which is the one outcome the caller who armed
            nothing is least likely to want; and it would do so silently, since
            a max-count artifact is indistinguishable from a correct one until
            its output is wrong.
    """
    spec_list = list(specs)
    maxima = symbolic_count_maxima(spec_list)
    if not maxima:
        return spec_list, None
    count = current_trip_count()
    if count is None:
        raise TripCountUnsetError(
            f"a kernel whose trip count is chosen at launch reached code "
            f"generation with no trip count armed on this thread (loop maxima "
            f"{maxima}). Wrap the call in "
            "`with torch_spyre.execution.trip_count.trip_count(k):`. Turning "
            "SPYRE_TRIP_COUNT_VARIANTS off does not help by itself: this "
            "wrapper was traced with the count left open, so it has no count "
            "to fall back to and must be re-traced with the caches cleared."
        )
    return substitute_trip_counts(spec_list, count), count
