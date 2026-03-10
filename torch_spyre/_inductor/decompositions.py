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

from contextlib import contextmanager

from typing import Optional, Union, Sequence, Callable, TypeVar
from typing_extensions import ParamSpec
import torch
from torch.utils import _pytree as pytree
import torch._decomp as decomp

from .constants import DEVICE_NAME
from .errors import Unsupported
from . import customops  # noqa: F401

# Dictionary for Spyre-specific decompositions
spyre_decompositions: dict = {}

# Exclude specific Inductor default decompositions on Spyre.
# Some Inductor decompositions do not work reliably on the Spyre backend yet.
# We disable them here and rely on implicit fallbacks to eager ops instead. Once
# the blocking issues are resolved, these exclusions can be removed.
spyre_decompositions_to_exclude = [
    # The default decomposition for torch.new_ones (defined in pytorch/torch/refs/__init__.py)
    # uses torch.full, which is not yet supported in Spyre eager mode.
    # See: https://github.com/torch-spyre/torch-spyre/issues/128#issuecomment-3576168221
    torch.ops.aten.new_ones,
]

# Dict for Spyre-specific decompositions to be registered via DispatchKey
spyre_decompositions_via_dispatchkey: dict = {}

# Module-level Library objects kept alive permanently so that the registered
# PrivateUse1 / AutogradPrivateUse1 kernels are never unregistered by garbage collector.
# (torch.library.Library uses weakref.finalize → m.reset() on GC, which would
# silently remove the kernels from the C++ dispatcher.)
_spyre_autograd_lib = None
_spyre_lib = None
_dispatchkey_kernels_registered = False

_T = TypeVar("_T")
_P = ParamSpec("_P")


def register_spyre_decomposition(
    ops: Union[torch._ops.OperatorBase, list],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register decompositions specifically for the Spyre compile-time decomposition table.
    These are active during torch.compile (make_fx tracing) when the Spyre decomposition
    table is in use.

    Use this decorator for ops that only need compile-time decomposition support and do
    NOT require eager-mode dispatch through the PyTorch dispatcher.  Examples: full, gt,
    lt, logical_not.

    For ops that need BOTH compile-time decomposition AND eager-mode dispatch (e.g. gelu,
    layer_norm, rms_norm, softplus), use @register_spyre_decompositions_via_dispatchkey
    alone — it now registers the function in both spyre_decompositions (compile path) and
    spyre_decompositions_via_dispatchkey (eager path).
    """
    return decomp.register_decomposition(ops, spyre_decompositions)


# Context manager that builds a per-compilation decomposition table for Spyre.
@contextmanager
def enable_spyre_decompositions(
    decomps: Optional[dict[torch._ops.OperatorBase, Callable]] = None,
):
    """
    CM that builds a per-compilation decomposition table for Spyre:
      - Creates a fresh copy of the base table (never mutates the original)
      - Merges Spyre-specific overrides from spyre_decompositions
      - Removes excluded ops (spyre_decompositions_to_exclude, fallback_ops)

    Thread-safe by construction: each call yields an independent dict, so no
    lock is needed.

    Args:
        decomps: Base decomposition table to build from. Maps operator overloads
            to their decomposition implementations. Defaults to PyTorch Inductor's
            global decomposition registry.
    """
    from torch._ops import OpOverload, OpOverloadPacket
    from torch_spyre.fallbacks import fallback_ops

    base = (
        decomps
        if decomps is not None
        else torch._inductor.decomposition.select_decomp_table()
    )

    # Fresh copy per compilation — the original dict is never modified.
    merged = dict(base)

    # Override with Spyre-specific implementations.
    merged.update(spyre_decompositions)

    # Remove ops that must fall back to eager or have known incompatibilities.
    def _remove_ops(ops_to_remove):
        for op in ops_to_remove:
            if isinstance(op, OpOverloadPacket):
                for overload_name in op.overloads():
                    merged.pop(getattr(op, overload_name), None)
            elif isinstance(op, OpOverload):
                merged.pop(op, None)

    _remove_ops(spyre_decompositions_to_exclude)
    _remove_ops(fallback_ops)

    yield merged


class _SpyreGetDecompFn:
    """
    Stable module-level callable that returns the merged Spyre decomposition table.

    Being a singleton (``_spyre_get_decomp_fn``), its object identity is constant
    across compilations.  This means ``functools.cache`` in upstream ``lazy_init``
    and ``_sfdp_init`` creates exactly **one** cache entry for all Spyre
    compilations, rather than a new entry per compilation (which would happen with
    an inline lambda that captures a fresh dict each time).

    The merged table is computed lazily on first call and then reused.  It is safe
    to cache because all inputs — the global Inductor decompositions table,
    ``spyre_decompositions``, ``fallback_ops``, and
    ``spyre_decompositions_to_exclude`` — are fixed after module load.

    Note: ``decompositions`` passed to ``compile_fx`` for AOT Autograd tracing is
    still the per-compilation fresh copy produced by ``enable_spyre_decompositions``.
    This singleton is only used for joint-graph pattern matching (SFDP etc.).
    """

    _decomps: Optional[dict] = None

    def __call__(self) -> dict:
        if self._decomps is None:
            from torch._ops import OpOverload, OpOverloadPacket
            from torch_spyre.fallbacks import fallback_ops

            base = torch._inductor.decomposition.select_decomp_table()
            merged = dict(base)
            merged.update(spyre_decompositions)

            def _remove_ops(ops_to_remove):
                for op in ops_to_remove:
                    if isinstance(op, OpOverloadPacket):
                        for overload_name in op.overloads():
                            merged.pop(getattr(op, overload_name), None)
                    elif isinstance(op, OpOverload):
                        merged.pop(op, None)

            _remove_ops(spyre_decompositions_to_exclude)
            _remove_ops(fallback_ops)
            self._decomps = merged
        return self._decomps


_spyre_get_decomp_fn = _SpyreGetDecompFn()


def _register_spyre_dispatchkey_kernels_permanently():
    """
    Permanently register PrivateUse1 / AutogradPrivateUse1 kernels for all ops
    in ``spyre_decompositions_via_dispatchkey``.

    This must be called once before any eager-mode dispatch can reach the Spyre
    kernels (typically from ``_SpyreImpl._lazy_init()``).  It is idempotent:
    subsequent calls are no-ops.

    The ``Library`` objects are stored in module-level globals so they are never
    garbage-collected (and therefore never unregistered from the C++ dispatcher).

    After registration ``OPWrapper.__call__`` uses ``torch.compiler.is_compiling()``
    to route dispatch: inside a ``torch.compile`` context the Spyre function is called
    directly; outside (eager mode) the pre-compiled wrapper is used.
    """
    global _spyre_autograd_lib, _spyre_lib, _dispatchkey_kernels_registered

    if _dispatchkey_kernels_registered:
        return

    from torch.library import Library, fallthrough_kernel

    _spyre_autograd_lib = Library("aten", "IMPL", "AutogradPrivateUse1")
    _spyre_lib = Library("aten", "IMPL", "PrivateUse1")

    for op, wrapper_cls in spyre_decompositions_via_dispatchkey.items():
        # Autograd key: fall through so that the PrivateUse1 kernel is reached.
        _spyre_autograd_lib.impl(op._name, fallthrough_kernel, allow_override=True)
        # PrivateUse1 key: the OPWrapper dispatches to spyre_fn.
        # allow_override=True because codegen_ops.py may have already registered a
        # generic torch.compile-based implementation for the same op; the OPWrapper
        # with the dedicated Spyre custom op should take precedence.
        _spyre_lib.impl(op._name, wrapper_cls, allow_override=True)

    _dispatchkey_kernels_registered = True


def register_spyre_decompositions_via_dispatchkey(
    ops: Union[torch._ops.OperatorBase, list],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register decompositions specifically for Spyre device via the PyTorch dispatcher
    This replaces the need for global patching of operations in order to enable them for
    eager mode.
    """

    def decomposition_decorator(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # Also register the raw function in spyre_decompositions so that compile-time
        # make_fx tracing (inside AOT Autograd) uses the Spyre implementation instead
        # of falling through to CompositeImplicitAutograd.  This removes the need to
        # apply @register_spyre_decomposition as a second decorator for ops that also
        # need eager-mode dispatch via the PrivateUse1 key.
        decomp.register_decomposition(ops, spyre_decompositions)(fn)

        class OPWrapper:
            def __init__(self, op, spyre_fn):
                self.op = op
                self.spyre_fn = spyre_fn
                # Pre-compile once so that repeated eager-mode calls reuse the
                # same compiled entry point rather than constructing a new
                # torch.compile wrapper on every invocation.
                self._compiled_fn = torch.compile(spyre_fn)

            def __call__(self, *args, **kwargs):
                # We are about to execute the op on spyre; inputs must be on spyre.
                if any(
                    isinstance(x, torch.Tensor)
                    and getattr(x.device, "type", None) != DEVICE_NAME
                    for x in (pytree.tree_leaves(args) + pytree.tree_leaves(kwargs))
                ):
                    raise RuntimeError(
                        "Spyre decomposition function called with inputs being on a different device!"
                    )

                # Inside a torch.compile context (make_fx tracing, Inductor
                # lowering, etc.) call the function directly — wrapping it in
                # another torch.compile call would be incorrect.
                if torch.compiler.is_compiling():
                    return self.spyre_fn(*args, **kwargs)
                else:
                    # Eager mode: use the pre-compiled wrapper.
                    return self._compiled_fn(*args, **kwargs)

        def register(op):
            spyre_decompositions_via_dispatchkey[op] = OPWrapper(op, fn)

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, ops)
        return fn

    return decomposition_decorator


@contextmanager
def enable_spyre_decompositions_via_dispatchkey():
    """
    Context manager that ensures the Spyre PrivateUse1 kernels are registered
    for the duration of a ``torch.compile`` call.

    Kernels are registered permanently in the C++ dispatcher by
    ``_register_spyre_dispatchkey_kernels_permanently()`` (idempotent).
    Once registered, ``OPWrapper.__call__`` uses ``torch.compiler.is_compiling()``
    to route dispatch: inside a ``torch.compile`` context the Spyre function is
    called directly; outside (eager mode) the pre-compiled wrapper is used.

    The CM is reentrant.
    """
    _register_spyre_dispatchkey_kernels_permanently()
    yield


@register_spyre_decomposition([torch.ops.spyre.compact])
def compact_decomp(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


# TODO (imaihal): Inductor applies constant folding to torch.full, which allocates
# a one-element Spyre tensor. This currently fails because Spyre does not handle
# single-element tensors well.
# Ref: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/fx_passes/joint_graph.py#L324-L335
#
# To avoid constant folding, we introduce a custom op `spyre::full` that runs
# torch.full on CPU and copies the result to Spyre. Remove this workaround once
# Spyre supports one-element tensors.
@register_spyre_decomposition([torch.ops.aten.full])
def full_decomp(
    size: list[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: Optional[bool] = None,
) -> torch.Tensor:
    assert layout in (torch.strided, None), f"doesn't support layout={layout}"
    assert not pin_memory, f"doesn't support pin_memory={pin_memory}"
    return torch.ops.spyre.full(size, fill_value, device, dtype=dtype)


@register_spyre_decomposition([torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Tensor_out])
def gt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement greaterthan in the backend compiler
    out_ge = torch.ge(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_ge, out_ne, out=out).to(dtype=torch.bool)


@register_spyre_decomposition([torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Tensor_out])
def lt_decomp(
    input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # TODO: Implement lessthan in the backend compiler
    out_le = torch.le(input, other).to(dtype=torch.float16)
    out_ne = torch.ne(input, other).to(dtype=torch.float16)
    return torch.mul(out_le, out_ne, out=out).to(dtype=torch.bool)


@register_spyre_decomposition([torch.ops.aten.logical_not])
def logical_not_decomp(input: torch.Tensor) -> torch.Tensor:
    # Currently falling back to torch.zeros_like for dtypes other than bool
    # This is needed until scalar False/0.0 or constant tensor [False]/[0.0] is supported
    if input.dtype is torch.bool:
        zero = torch.ne(input, input)
    else:
        zero = torch.zeros_like(input)
    return torch.eq(input, zero)


###############################################################################################
##                           Functions requiring dispatch keys                               ##
###############################################################################################
# @register_spyre_decompositions_via_dispatchkey is sufficient for these ops: it registers
# both the PrivateUse1 kernel (for eager-mode dispatch) and an entry in spyre_decompositions
# (for compile-time make_fx tracing, preventing CIA from running).
@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.rms_norm.default])
def spyre_rms_norm(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_rms_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )

    # TODO: limitation with mean on dim=-1, transpose for now to avoid
    # https://github.com/torch-spyre/torch-spyre/issues/632
    input = input.transpose(-1, -2).contiguous()
    eps_tensor = torch.ops.spyre.full(
        input.shape, eps, dtype=torch.float16, device="spyre"
    )
    rsqrt_inp = (
        torch.rsqrt(torch.mean(input * input, dim=-2, keepdim=True)) + eps_tensor
    )
    output = (input * rsqrt_inp).transpose(-1, -2).contiguous()
    if weight is not None:
        output = output * weight
    return output


@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre_layer_norm: only supports spyre device with normalized_shape of length 1, "
            f"got device={input.device.type}, normalized_shape={normalized_shape}"
        )
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.gelu.default])
def spyre_gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    return torch.ops.spyre.gelu(input, approximate)


@register_spyre_decompositions_via_dispatchkey([torch.ops.aten.softplus.default])
def spyre_softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    return torch.ops.spyre.softplus(input, beta, threshold)
