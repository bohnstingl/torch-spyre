# Copyright 2025-2026 The Torch-Spyre Authors.
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

from collections.abc import Sequence
from typing import Any

import torch
from torch_spyre._C import (
    SymbolicArg,
    launch_jobplan,
    prepare_kernel,
    register_kernel_provenance,
)
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.kernel_provenance import KernelProvenanceDescriptor
from torch_spyre._inductor.profiler_event import (
    format_kernel_provenance_event_name,
)
from torch_spyre.profiler._ffdc import (
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    with_ffdc,
)

logger = get_inductor_logger("kernel_runner")


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    @with_ffdc(CATEGORY_UNIMPLEMENTED, logger, code_dir_attr=None)
    def run(self, *args, **kw_args):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains"
            f" unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    """Kernel runner for a compiled SDSC bundle.

    The jobplan handle is initialised lazily on first call to :meth:`run`.
    This avoids calling ``prepare_kernel`` (which requires a live C++
    RuntimeContext) in the compiling process; the context is only guaranteed
    to be available on the process that actually launches the kernel.

    A runner given a spec tree whose ``for_each_tile`` trip count was left
    symbolic by the frontend (``config.spyre_trip_count_variants``) additionally
    becomes *variant-capable*: it keeps one jobplan per trip count, substitutes
    the count into the retained tree to build the ones it does not have, and
    picks between them from the ambient count at launch. See
    ``execution/trip_count.py``.
    """

    def __init__(
        self,
        name: str,
        code_dir: str,
        kernel_provenance: KernelProvenanceDescriptor | None = None,
        specs: Sequence[Any] | None = None,
        pool_size: int = 0,
        base_count: int | None = None,
    ):
        self.kernel_name = name
        self.code_dir = code_dir
        self.kernel_provenance = kernel_provenance
        self.profiler_event_name: str | None
        self._jobplan = None  # initialised lazily, not pickled

        from .trip_count import register_runner, variant_capable

        self._max_count = variant_capable(specs)
        # Held only when variant-capable, so an ordinary kernel does not keep its
        # spec tree alive for the process's lifetime.
        self._specs = list(specs) if self._max_count is not None else None
        self._pool_size = pool_size
        self._code_dirs: dict[int, str] = {}
        self._jobplans: dict[int, Any] = {}
        if self._max_count is not None:
            # ``code_dir`` is not a default to fall back to -- it is simply the
            # first variant, the one the ambient count at compile time asked for
            # (see async_compile.sdsc). Recording it under that count is what
            # keeps the compile that just happened from being repeated on the
            # first launch.
            if base_count is not None:
                self._code_dirs[base_count] = code_dir
            register_runner(self)
            logger.debug(
                "%s is variant-capable up to trip count %d (compiled for %s)",
                name,
                self._max_count,
                base_count,
            )

        if kernel_provenance is None:
            self.profiler_event_name = None
        else:
            self.profiler_event_name = format_kernel_provenance_event_name(
                kernel_provenance
            )
            # Rejection is intentionally fail-open: C++ warns and counts
            # conflicts while the key-bearing name remains the compatibility
            # join.
            register_kernel_provenance(
                self.profiler_event_name,
                list(kernel_provenance.debug_handle_ids),
            )

    def _prepare(self, code_dir: str):
        logger.debug("Initialising jobplan for %s from %s", self.kernel_name, code_dir)
        # _lazy_init() ensures the C++ RuntimeContext is initialised before
        # prepare_kernel(), which calls into JobPlanBuilder/getDefaultStream().
        torch.spyre._impl._lazy_init()
        spyrecode_dir = code_dir + "/spyreCodeDir"
        if self.profiler_event_name is None:
            return prepare_kernel(spyrecode_dir)
        with torch.profiler.record_function(f"prepare_kernel:{self.kernel_name}"):
            return prepare_kernel(
                spyrecode_dir,
                profiler_name=self.profiler_event_name,
            )

    @property
    def jobplan(self):
        if self._jobplan is None:
            self._jobplan = self._prepare(self.code_dir)
        return self._jobplan

    # -- variants ----------------------------------------------------------

    @property
    def max_trip_count(self) -> int | None:
        """The largest launchable count, or ``None`` if this runner has no variants.

        The trace fixed a maximum -- every descriptor is sized for it -- but no
        particular count, so this is a ceiling rather than a default.
        """
        return self._max_count

    def has_variant(self, count: int) -> bool:
        return count in self._code_dirs

    def prepare_variant(self, count: int):
        """Emit ``count``'s bundle without compiling it. See ``trip_count``."""
        from .trip_count import prepare_variant

        return prepare_variant(self.kernel_name, self._specs, count, self._pool_size)

    def adopt_variant(self, count: int, code_dir: str) -> None:
        self._code_dirs[count] = code_dir

    def build_variant(self, count: int) -> str:
        """Compile ``count``'s variant from the retained spec tree."""
        from .trip_count import build_variant

        code_dir = build_variant(self.kernel_name, self._specs, count, self._pool_size)
        self.adopt_variant(count, code_dir)
        return code_dir

    def _jobplan_for(self, count: int):
        if count not in self._code_dirs:
            self.build_variant(count)
        jobplan = self._jobplans.get(count)
        if jobplan is None:
            jobplan = self._prepare(self._code_dirs[count])
            self._jobplans[count] = jobplan
        return jobplan

    def _select_jobplan(self):
        """The jobplan to launch: the only one, or the ambient count's."""
        if self._max_count is None:
            return self.jobplan, self.code_dir

        from .trip_count import TripCountUnsetError, current_trip_count

        count = current_trip_count()
        if count is None:
            raise TripCountUnsetError(
                f"{self.kernel_name} was traced with its trip count left to "
                f"launch (maximum {self._max_count}) but no trip count is armed "
                "on this thread. Wrap the call in "
                "`with torch_spyre.execution.trip_count.trip_count(k):`. There "
                "is deliberately no default: a caller passing max-size buffers "
                f"would then silently run up to {self._max_count} iterations "
                "over rows it never wrote."
            )
        return self._jobplan_for(count), self._code_dirs[count]

    @with_ffdc(CATEGORY_RUNTIME_LAUNCH, logger)
    def run(self, *args, symbolic_args: list[SymbolicArg] | None = None, **kw_args):
        jobplan, code_dir = self._select_jobplan()
        logger.info("RUN: %s %s", self.kernel_name, code_dir)
        with torch.profiler.record_function(f"launch_jobplan:{self.kernel_name}"):
            if symbolic_args:
                launch_jobplan(jobplan, args, symbolic_args)
            else:
                launch_jobplan(jobplan, args)
