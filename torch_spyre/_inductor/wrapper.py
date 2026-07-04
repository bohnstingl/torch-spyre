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

from typing import Optional

import sympy
from torch._inductor.codegen.wrapper import (
    BufferLike,
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.utils import ValueWithLineMap
from torch._inductor.virtualized import V
from torch._inductor.sizevars import SizeVarAllocator

from .ir import FixedTiledLayout
from .constants import SEGMENT_SIZE, INTERMEDIATES_SEGMENT


class SpyrePythonWrapperCodegen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
            V.graph.sizevars, SizeVarAllocator
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from sympy import sympify
                from torch_spyre._inductor.op_spec import TensorArg, OpSpec, UnimplementedOp, LoopSpec, spyre_constant_tensor, IndirectAccess
                from torch_spyre.execution.async_compile import SpyreAsyncCompile
                from torch_spyre._C import DataFormats, SpyreTensorLayout, spyre_empty_with_layout, set_spyre_tensor_layout
                import subprocess
            """,
            strip=True,
        )
        # Device-aware reinterpret_tensor: Spyre tensors use the _C binding;
        # CPU buffers (from fused host kernels feeding CPU custom ops like
        # spyre_rotary_cpu) delegate to torch's _reinterpret_tensor. The bare
        # _C binding static_casts to SpyreTensorImpl and crashes with
        # std::bad_alloc on non-Spyre input. See reinterpret_shim.py.
        self.header.writeline(
            "from torch_spyre._inductor.reinterpret_shim import reinterpret_tensor as reinterpret_tensor"
        )
        self.header.writeline(
            "from torch_spyre._C import reinterpret_tensor_with_layout"
        )
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")

    def generate(self, is_inference):
        """Override to add pool allocation/deallocation around kernel calls."""
        result_tuple = super().generate(is_inference)
        wrapper_value_with_linemap, kernel_decls = result_tuple

        pool_size = getattr(V.graph, "pool_size", 0)
        if pool_size > 0:
            wrapper_str = str(wrapper_value_with_linemap.value)

            # Inject pool allocation before kernel calls and cleanup before return.
            lines = wrapper_str.split("\n")

            # Add `del _pool` before `return (` statement.
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line.startswith("return ("):
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    lines.insert(i, " " * indent + "del _pool")
                    break

            # Add pool allocation before the FIRST line that references `_pool`.
            # That is normally the first SDSC kernel `.run(...)`, but with the
            # pool-resident buffer views emitted by `make_buffer_allocation`
            # (`reinterpret_tensor_with_layout(_pool, ...)` for pool buffers a
            # downstream FallbackKernel reads), a `_pool` reference can appear
            # BEFORE the first `.run(`. Injecting only before `.run(` then
            # leaves those views referencing an undefined `_pool`
            # (`UnboundLocalError`). Match either form.
            pool_alloc_code = self.allocate_pool()
            for i, line in enumerate(lines):
                if ".run(" in line or "reinterpret_tensor_with_layout(_pool" in line:
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i, " " * indent + pool_alloc_code)
                    break

            wrapper_str = "\n".join(lines)
            wrapper_value_with_linemap = ValueWithLineMap(
                value=wrapper_str, line_map=wrapper_value_with_linemap.line_map
            )

        return (wrapper_value_with_linemap, kernel_decls)

    def make_buffer_allocation(self, buffer: BufferLike):
        layout = buffer.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            return super().make_buffer_allocation(buffer)

        name = buffer.get_name()
        codegen_shape_tuple = self.codegen_python_shape_tuple(tuple(layout.size))
        codegen_stride_tuple = self.codegen_python_shape_tuple(tuple(layout.stride))

        if "pool" in layout.allocation:
            # Pool-resident: the SDSC producer kernel wrote this buffer into
            # `_pool` at a fixed byte offset. When it survives buffer removal
            # (because a downstream FallbackKernel needs a Python handle — see
            # spyre_kernel.remove_kernel_local_buffers), materialize a
            # reinterpret view into `_pool` at that offset rather than a fresh
            # allocation (which would point at the wrong storage and feed the
            # FallbackKernel garbage). `_pool` is 1-D uint8 (1 elem = 1 byte),
            # so offset_increment is the byte offset relative to
            # INTERMEDIATES_SEGMENT. Pass `dtype=layout.dtype` so the view is
            # tagged fp16/fp32/... instead of inheriting _pool's uint8.
            byte_offset = layout.allocation["pool"] - INTERMEDIATES_SEGMENT
            return (
                f"{name} = reinterpret_tensor_with_layout("
                f"_pool, "
                f"{codegen_shape_tuple}, "
                f"{codegen_stride_tuple}, "
                f"{byte_offset}, "
                f"{layout.device_layout!r}, "
                f"dtype={layout.dtype})"
            )

        out = (
            f"{name} = spyre_empty_with_layout("
            f"{codegen_shape_tuple}, "
            f"{codegen_stride_tuple}, "
            f"{layout.dtype}, "
            f"{layout.device_layout!r})"
        )

        return out

    def generate_const_tensor_fallback(self, node):
        value = node.constant_args[0]
        dtype = node.layout.dtype
        device = node.layout.device
        self.writeline(
            f'{node.get_name()} = spyre_constant_tensor({value}, torch.device("{device}"), {dtype})'
        )

    def _is_pool_buffer(self, buffer: BufferLike) -> bool:
        layout = buffer.get_layout()
        return isinstance(layout, FixedTiledLayout) and "pool" in layout.allocation

    def codegen_free_buffer(self, buffer: BufferLike) -> None:
        if not self._is_pool_buffer(buffer):
            super().codegen_free_buffer(buffer)

    def make_buffer_reuse(self, old: BufferLike, new: BufferLike, delete_old: bool):
        assert old.get_dtype() == new.get_dtype()
        old_name = old.get_name()
        new_name = new.get_name()
        del_line = ";"
        if old_name not in V.graph.get_output_names() and delete_old:
            del_line = f"; {self.make_buffer_free(old)}"

        if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
            return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)

        new_stl = new.get_layout().device_layout
        reinterpret_view = f"reinterpret_tensor_with_layout({old_name}, {new.get_size()}, {new.get_stride()}, 0, {new_stl!r})"
        return f"{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse"

    def allocate_pool(self):
        """Allocate the intermediate pool."""
        pool_size_bytes = getattr(V.graph, "pool_size", SEGMENT_SIZE)
        pool_size_sticks = (pool_size_bytes + 127) // 128
        return (
            f"_pool = spyre_empty_with_layout("
            f"({pool_size_sticks},), (1,), "
            f"torch.uint8, SpyreTensorLayout(device_size=[{pool_size_sticks}, 1, 1], "
            f"stride_map=[1, 1, 1], device_dtype=DataFormats.SENINT8))"
        )


def noop_simplify_loops_impl(
    self, index_vars: list[sympy.Symbol], sizes, index_formulas
):
    """
    This is a noop implementation of SizeVarAllocator._simplify_loops_impl.

    We do this because the memory layout of tensors on the Spyre device is not
    entirely visible to Inductor.  Therefore Inductor's understanding of which
    tensor dimensions are actually contiguous is not accurate.
    """
    return sizes, lambda x: x, lambda x: x
