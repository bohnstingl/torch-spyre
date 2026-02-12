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

# Owner(s): ["module: cpp"]

import pathlib
import yaml
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_methods_invocations import op_db

from _inductor.utils_inductor import compare_with_cpu


class TestOps(TestCase):
    def __init__(self, method_name="runTest", methodName="runTest"):
        super().__init__(method_name, methodName)
        declarations_path = (
            pathlib.Path(__file__).parent.parent
            / "codegen"
            / "outputs"
            / "GeneratedDeclarations.yaml"
        )
        declarations_path.resolve()

        with declarations_path.open() as f:
            self.declarations = yaml.safe_load(f)

        # NOTE: needs to be at most 1e-3
        self.rtol = 1e-1
        self.atol = 1e-1
        self.dtype = torch.float16

        # TODO: The tensor size was changed (from 3, 5, 7 respectively) to avoid padding in the stick dimension.
        #   Once we have proper padding to stack handled, these values should be changed back
        self.mm_a = 67
        self.mm_b = 256
        self.mm_c = 128
        torch.random.manual_seed(42)

    def test_inplace_fill_scalar(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype, device="spyre")
        x.fill_(5.0)
        x_actual = x.cpu()
        x_expected = torch.tensor([5.0, 5.0, 5.0], dtype=self.dtype)
        torch.testing.assert_close(x_expected, x_actual, rtol=self.rtol, atol=self.atol)

    def test_copy_1d_padded_to_stick(self):
        x = torch.tensor([1, 2, 3], dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_2d_padded_to_stick(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_3d_padded_to_stick(self):
        x = torch.tensor(
            [[[1, -2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=self.dtype,
        )
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_4d_padded_to_stick(self):
        x = torch.rand(2, 2, 2, 3, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_6d_padded_to_stick(self):
        x = torch.rand(1, 3, 5, 2, 4, 62, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_5d_padded_to_stick(self):
        x = torch.rand(1, 2, 3, 4, 5, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_4d_padded(self):
        x = torch.rand(2, 2, 2, 120, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_3d_padded(self):
        x = torch.rand(2, 2, 72, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_2d_padded(self):
        x = torch.rand(2, 205, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_1d_padded(self):
        x = torch.rand(511, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_1d(self):
        x = torch.rand(256, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_2d(self):
        x = torch.rand(256, 128, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_3d(self):
        x = torch.rand(256, 128, 512, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_4d(self):
        x = torch.rand(2, 6, 3, 128, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_5d(self):
        x = torch.rand(4, 8, 3, 64, 256, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_copy_6d(self):
        x = torch.rand(4, 8, 16, 12, 64, 128, dtype=self.dtype)
        y = x.to("spyre").to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    @unittest.skip("View tensors do not have SpyreTensorImpl")
    def test_t_1d(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.t().to("cpu")
        torch.testing.assert_close(y, x.t(), rtol=self.rtol, atol=self.atol)

    @unittest.skip("Swapping stick dimension is unsupported in new DCI")
    def test_t_2d(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.t().to("cpu")
        torch.testing.assert_close(y, x.t(), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Implement permute/transpose properly on eager")
    def test_transpose_2d(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.transpose(0, 1).to("cpu")
        torch.testing.assert_close(y, x.transpose(0, 1), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Implement permute/transpose properly on eager")
    def test_transpose_3d(self):
        x = torch.tensor(
            [[[1, -2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=self.dtype,
        )
        x_spyre = x.to("spyre")
        y = x_spyre.transpose(0, 1).to("cpu")
        torch.testing.assert_close(y, x.transpose(0, 1), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Implement permute/transpose properly on eager")
    def test_permute_2d(self):
        x = torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = x_spyre.permute(1, 0).to("cpu")
        torch.testing.assert_close(y, x.permute(1, 0), rtol=self.rtol, atol=self.atol)

    def test_bool(self):
        dtype = torch.bool
        x = torch.randint(0, 2, (2, 64), dtype=dtype)
        x_spyre = x.to("spyre")
        y = torch.randint(0, 2, (2, 64), dtype=dtype)
        y_spyre = y.to("spyre")
        result = torch.eq(x_spyre, y_spyre).cpu()
        torch.testing.assert_close(result, torch.eq(x, y))

    def test_eq(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        y = torch.tensor([0, -2, 4], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        # FIXME: equal is currently returning back the same dtype as the original tensor, we need to have this return a bool
        actual = (x_spyre == y_spyre).cpu().bool()
        torch.testing.assert_close(actual, x == y, rtol=self.rtol, atol=self.atol)

    def test_ge(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        y = torch.tensor([0, -2, 4], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        # FIXME: equal is currently returning back the same dtype as the original tensor, we need to have this return a bool
        actual = (x_spyre >= y_spyre).cpu().bool()
        torch.testing.assert_close(actual, x >= y, rtol=self.rtol, atol=self.atol)

    def test_abs(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.abs(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.abs(x), rtol=self.rtol, atol=self.atol)

    def test_relu(self):
        x = torch.tensor([1, -2, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.relu(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.relu(x), rtol=self.rtol, atol=self.atol)

    def test_exp(self):
        x = torch.tensor([-10, -1, 0, 1, 10], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.exp(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.exp(x), rtol=self.rtol, atol=self.atol)

    def test_exp_transpose(self):
        x = torch.tensor([[-10, -1, 0, 1, 10], [1, 2, 3, 4, 5]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.exp(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.exp(x), rtol=self.rtol, atol=self.atol)

    def test_log(self):
        x = torch.tensor([0.1, 1, 10, 100], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.log(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.log(x), rtol=self.rtol, atol=self.atol)

    def test_reciprocal(self):
        x = torch.tensor([-2, 1, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.reciprocal(x_spyre).to("cpu")
        torch.testing.assert_close(
            y, torch.reciprocal(x), rtol=self.rtol, atol=self.atol
        )

    def test_sigmoid(self):
        x = torch.tensor([-2, 1, 3], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.sigmoid(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.sigmoid(x), rtol=self.rtol, atol=self.atol)

    def test_sqrt(self):
        x = torch.tensor([0, 1, 2.25, 4, 10000], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.sqrt(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.sqrt(x), rtol=self.rtol, atol=self.atol)

    def test_tanh(self):
        x = torch.tensor([-2, -1, 0, 1, 2], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.tanh(x_spyre).to("cpu")
        torch.testing.assert_close(y, torch.tanh(x), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Needs more debug")
    def test_clone(self):
        x = torch.tensor([-2, -1, 0, 1, 2], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y = torch.clone(x_spyre).to("cpu")
        torch.testing.assert_close(y, x, rtol=self.rtol, atol=self.atol)

    def test_add_Tensor(self):
        x = torch.tensor([1, 2, 3], dtype=self.dtype)
        y = torch.tensor([4, 5, 6], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.add(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.add(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.expectedFailure
    def test_add_Scalar(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=self.dtype)
        y = 5
        x_spyre = x.to("spyre")
        z = (x_spyre + y).to("cpu")
        torch.testing.assert_close(z, x + y, rtol=self.rtol, atol=self.atol)

    @unittest.skip("xfail: Swapping stick dimension is unsupported in new DCI")
    def test_add_Tensor_transpose(self):
        x = torch.arange(8, dtype=self.dtype).view(2, 4)
        y = torch.arange(8, dtype=self.dtype).view(4, 2) * 10
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z_01 = torch.add(x_spyre, y_spyre.t().contiguous()).to("cpu")
        z_10 = torch.add(x_spyre.t().contiguous(), y_spyre).to("cpu")
        torch.testing.assert_close(
            z_01, torch.add(x, y.t()), rtol=self.rtol, atol=self.atol
        )
        torch.testing.assert_close(
            z_10, torch.add(x.t(), y), rtol=self.rtol, atol=self.atol
        )

    def test_sub(self):
        x = torch.tensor([10, 20, 3], dtype=self.dtype)
        y = torch.tensor([4, 5, 6], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.subtract(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(
            z, torch.subtract(x, y), rtol=self.rtol, atol=self.atol
        )

    def test_mul(self):
        x = torch.tensor([1, 0, -3], dtype=self.dtype)
        y = torch.tensor([4, 5, 6], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mul(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mul(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_ab_bc(self):
        x = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_a, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_ac_cb(self):
        x = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_a, self.mm_c
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_b
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Debug accuracy error")
    def test_mm_ba_ac(self):
        x = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_b, self.mm_a
        )
        y = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_a, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_bc_ca(self):
        x = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        y = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: Debug accuracy error")
    def test_mm_ca_ab(self):
        x = torch.randn(self.mm_a * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_a
        )
        y = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_a, self.mm_b
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    def test_mm_cb_ba(self):
        x = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_c, self.mm_b
        )
        y = torch.randn(self.mm_a * self.mm_b, dtype=self.dtype).view(
            self.mm_b, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.mm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.mm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: bmm.out not implemented yet in eager")
    def test_bmm_ab_bc(self):
        B = 1
        x = torch.randn(B * self.mm_a * self.mm_b, dtype=self.dtype).view(
            B, self.mm_a, self.mm_b
        )
        y = torch.randn(B * self.mm_b * self.mm_c, dtype=self.dtype).view(
            B, self.mm_b, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.bmm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.bmm(x, y), rtol=self.rtol, atol=self.atol)

    @unittest.skip("TODO: bmm.out not implemented yet in eager")
    def test_bmm_cb_ba(self):
        B = 1
        x = torch.randn(B * self.mm_c * self.mm_b, dtype=self.dtype).view(
            B, self.mm_c, self.mm_b
        )
        y = torch.randn(B * self.mm_b * self.mm_a, dtype=self.dtype).view(
            B, self.mm_b, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.bmm(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(z, torch.bmm(x, y), rtol=self.rtol, atol=self.atol)

    def test_matmul_ab_bc(self):
        B = 1
        x = torch.randn(B * self.mm_a * self.mm_b, dtype=self.dtype).view(
            B, self.mm_a, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_c, dtype=self.dtype).view(
            self.mm_b, self.mm_c
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.matmul(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(
            z, torch.matmul(x, y), rtol=self.rtol, atol=self.atol
        )

    def test_matmul_cb_ba(self):
        B = 1
        x = torch.randn(B * self.mm_c * self.mm_b, dtype=self.dtype).view(
            B, self.mm_c, self.mm_b
        )
        y = torch.randn(self.mm_b * self.mm_a, dtype=self.dtype).view(
            self.mm_b, self.mm_a
        )
        x_spyre = x.to("spyre")
        y_spyre = y.to("spyre")
        z = torch.matmul(x_spyre, y_spyre).to("cpu")
        torch.testing.assert_close(
            z, torch.matmul(x, y), rtol=self.rtol, atol=self.atol
        )

    def test_sdpa_comp(self):
        # num_queries_per_kv = 4
        num_queries_per_kv = 1
        working_precision = torch.float16
        scale = torch.tensor(0.08838834764831843, dtype=working_precision)

        query = torch.randn([1, 16, 32, 128], dtype=working_precision)
        # key = torch.randn([1, 131072, 8, 128], dtype=working_precision)
        # value = torch.randn([1, 131072, 8, 128], dtype=working_precision)
        key = torch.randn([1, 131072, 32, 128], dtype=working_precision)
        value = torch.randn([1, 131072, 32, 128], dtype=working_precision)
        block_mask = (
            torch.randn([1, 32, 16, 131072], dtype=working_precision) > 0.0
        ).to(torch.bool)

        def _attn(q, k, v, b, s):
            # Handle grouped-query attention
            if num_queries_per_kv > 1:
                # Repeat KV heads to match query heads
                k = k.repeat_interleave(num_queries_per_kv, dim=2)
                v = v.repeat_interleave(num_queries_per_kv, dim=2)

            # Transpose for matmul
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Compute Q @ K^T
            attn_scores = torch.matmul(
                q.to(working_precision), k.to(working_precision).transpose(-2, -1)
            )

            # Scale
            attn_scores = (attn_scores * s).to(working_precision)

            if b is not None:
                attn_scores = attn_scores.masked_fill(b, -float("inf"))

            # Softmax
            attn_weights = torch._safe_softmax(attn_scores, dim=-1)

            # Compute attention output
            attn_output = torch.matmul(attn_weights.to(q.dtype), v.to(q.dtype))

            # Transpose back
            attn_output = attn_output.transpose(1, 2).squeeze(0)

            return attn_output

            return attn_scores

        compare_with_cpu(_attn, *[query, key, value, block_mask, scale])

    def test_sdpa_block(self):
        working_precision = torch.float16
        kv_cache = torch.randn([2, 8192, 16, 8, 128], dtype=working_precision)
        block_table = torch.tensor([[4, 5, 443], [1, 3, 2]], dtype=torch.int32)
        seq_lens = torch.tensor([32, 40], dtype=torch.int32)
        causal_mask = torch.tensor(
            [
                [
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            ],
            dtype=torch.bool,
        )
        slot_mapping = torch.tensor(
            [88, 89, 90, 91, 92, 93, 94, 95, 32, 33, 34, 35, 36, 37, 38, 39],
            dtype=torch.float32,
        )
        query_start_loc = torch.tensor([0, 8, 16], dtype=torch.float32)
        block_size = 16
        num_heads = 32

        def _block_creation(kv, bt, sq, cm, slm, qs):
            num_seqs = bt.shape[0]
            num_blocks, block_size, num_kv_heads, head_size = kv.shape[1:]

            num_tokens_per_sequence = [qs[i + 1] - qs[i] for i in range(len(qs) - 1)]
            max_num_tokens_per_sequence = torch.max(
                torch.stack(num_tokens_per_sequence)
            )

            # Initialize output tensor
            block_mask = torch.ones(
                num_seqs * max_num_tokens_per_sequence,
                num_blocks * block_size,
                dtype=torch.bool,
                device=kv.device,
            )

            # Gather tokens for each sequence
            for seq_idx in range(num_seqs):
                seq_remainder = sq[seq_idx] % block_size
                for bl_ind, bl in enumerate(bt[seq_idx]):
                    seq_len_curr_block = min(
                        sq[seq_idx] - bl_ind * block_size, block_size
                    )
                    max_bl_ind = (sq[seq_idx] // block_size) + (
                        1 if sq[seq_idx] % block_size else 0
                    )
                    if bl_ind < max_bl_ind:
                        block_mask[
                            seq_idx * max_num_tokens_per_sequence : (seq_idx + 1)
                            * max_num_tokens_per_sequence,
                            bl * block_size : bl * block_size + seq_len_curr_block,
                        ] = 0

                if cm is not None:
                    for token_nr in range(num_tokens_per_sequence[seq_idx] - 1):
                        slot_for_write = slm[qs[seq_idx] + 1 + token_nr :]
                        block_mask[
                            seq_idx * max_num_tokens_per_sequence + token_nr,
                            slot_for_write,
                        ] = 1

            block_mask = block_mask.unsqueeze(0).unsqueeze(0)
            block_mask = block_mask.expand(-1, num_heads, -1, -1)

            return block_mask

        compare_with_cpu(
            _block_creation,
            *[
                kv_cache,
                block_table,
                seq_lens,
                causal_mask,
                slot_mapping,
                query_start_loc,
            ],
        )

    def test_sdpa_KVupdate(self):
        working_precision = torch.float16
        key = torch.randn([2, 8, 128], dtype=working_precision)
        value = torch.randn([2, 8, 128], dtype=working_precision)
        kv_cache = torch.randn([2, 8192, 16, 8, 128], dtype=working_precision)
        slot_mapping = torch.tensor([95, 39], dtype=torch.int64)
        block_size = 16

        def _block_creation(k, v, kv, slm):
            num_tokens = k.shape[0]

            # Convert slot indices to block indices and offsets
            block_indices = slm // block_size
            block_offsets = slm % block_size

            # Get key and value caches
            key_cache = kv[0]
            value_cache = kv[1]

            # Write keys and values using advanced indexing
            for i in range(num_tokens):
                block_idx = block_indices[i].item()
                offset = block_offsets[i].item()
                key_cache[block_idx, offset] = k[i]
                value_cache[block_idx, offset] = v[i]

            return kv, key_cache, value_cache

        compare_with_cpu(_block_creation, *[key, value, kv_cache, slot_mapping])

    @unittest.skip("TODO: mean.out not implemented in eager mode")
    def test_mean(self):
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=self.dtype)
        x_spyre = x.to("spyre")
        y0 = torch.mean(x_spyre, dim=[0]).to("cpu")
        y1 = torch.mean(x_spyre, dim=[1]).to("cpu")
        y0_keepdim = torch.mean(x_spyre, dim=[0], keepdim=True).to("cpu")
        torch.testing.assert_close(
            y0, torch.mean(x, dim=[0]), rtol=self.rtol, atol=self.atol
        )
        torch.testing.assert_close(
            y1, torch.mean(x, dim=[1]), rtol=self.rtol, atol=self.atol
        )
        torch.testing.assert_close(
            y0_keepdim,
            torch.mean(x, dim=[0], keepdim=True),
            rtol=self.rtol,
            atol=self.atol,
        )

    def test_sum(self):
        x = torch.arange(0, 64, dtype=self.dtype).unsqueeze(0).repeat(3, 1)
        x_spyre = x.to("spyre")
        y0 = torch.sum(x_spyre, dim=[0]).to("cpu")
        torch.testing.assert_close(
            y0, torch.sum(x, dim=[0]), rtol=self.rtol, atol=self.atol
        )

    def test_softmax(self):
        x = torch.arange(0, 64, dtype=self.dtype).unsqueeze(0).repeat(3, 1)
        x_spyre = x.to("spyre")
        y1 = torch.softmax(x_spyre, dim=1).to("cpu")
        torch.testing.assert_close(
            y1, torch.softmax(x, dim=1), rtol=self.rtol, atol=self.atol
        )

    @unittest.skip("TODO: Needs more debug")
    def test_all_ops(self):
        def test_op(declaration):
            op_handle = getattr(torch.ops.aten, declaration["operator_name"])
            if declaration["overload_name"]:
                try:
                    op_handle = getattr(op_handle, declaration["overload_name"])
                except AttributeError:
                    pass

            op_info = [op for op in op_db if op.name == declaration["name"]]

            close = True
            if op_info:
                op_info = op_info[0]
                sample_inputs = list(
                    op_info.sample_inputs(device="cpu", dtype=torch.float16)
                )
                for s in sample_inputs:
                    sample_input = [
                        s.input,
                        *s.args[: len(declaration["arguments"]) - 1],
                    ]
                    try:
                        outputs_cpu = op_handle(*sample_input)

                        sample_input_spyre = [
                            s_.to("spyre") if isinstance(s_, torch.Tensor) else s_
                            for s_ in sample_input
                        ]
                        outputs_spyre = op_handle(*sample_input_spyre)

                        for j in range(len(outputs_cpu)):
                            close_ = torch.allclose(
                                outputs_cpu[j],
                                outputs_spyre[j].to("cpu"),
                                rtol=self.rtol,
                                atol=self.atol,
                            )
                            if not close_:
                                close = False
                                print(
                                    f"spyre output is different for {declaration['operator_name']}"
                                )

                        # check if something happens to inputs as well
                        for j in range(len(sample_input)):
                            if isinstance(sample_input[j], torch.Tensor):
                                close_ = torch.allclose(
                                    sample_input[j],
                                    sample_input_spyre[j].to("cpu"),
                                    rtol=self.rtol,
                                    atol=self.atol,
                                )
                                if not close_:
                                    close = False
                                    print(
                                        f"spyre inputs changed after operation for {declaration['operator_name']}"
                                    )

                    except Exception:
                        print(f"Could not run test for {declaration['operator_name']}")
            else:
                print(f"Could not find op_info for {declaration['operator_name']}")

            return close

        for dec in self.declarations:
            test_op(dec)


if __name__ == "__main__":
    run_tests()
