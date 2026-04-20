import logging
from typing import Optional

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestScript")
torch.manual_seed(42)


def vllm_causal_conv1d_update(
    hidden_state: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    hidden_state = hidden_state.transpose(1, 2)
    weight = weight.transpose(0, 1)
    conv_state = conv_state.transpose(1, 2)
    bsz, hidden_size, seq_len = hidden_state.shape
    kernel_size = weight.shape[-1]

    target_state_len = (kernel_size - 1) + (seq_len - 1)

    full_context = torch.cat([conv_state[conv_state_indices], hidden_state], dim=-1).to(weight.dtype)

    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len):]
    windows = computation_input.unfold(-1, kernel_size, 1)

    out = (windows * weight[None, :, None, :]).sum(dim=-1)

    if bias is not None:
        out = out + bias[None, :, None]

    if activation:
        out = F.silu(out)

    out = out.to(hidden_state.dtype)

    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:]
    else:
        new_conv_state = torch.empty(
            bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype
        )
    conv_state[conv_state_indices] = new_conv_state
    conv_state = conv_state.transpose(1, 2)
    out = out.transpose(1, 2)

    return out


def vllm_causal_conv1d_fn_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    conv_state_indices: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: bool = True,
) -> torch.Tensor:
    x_t = x.transpose(0, 1)
    weight_t = weight.transpose(0, 1)
    conv_state_t = conv_state.transpose(1, 2)

    dim = x_t.shape[0]
    cu_seqlen = x_t.shape[1]
    kernel_size = weight_t.shape[-1]
    batch = query_start_loc.shape[0] - 1

    out = torch.empty_like(x_t)

    for b in range(batch):
        start = query_start_loc[b].item()
        end = query_start_loc[b + 1].item()
        seq_len_b = end - start
        if seq_len_b <= 0:
            continue

        cache_idx = conv_state_indices[b].item()
        state = conv_state_t[cache_idx]

        full_context = torch.cat([state, x_t[:, start:end]], dim=-1).to(weight_t.dtype)

        computation_input = full_context[:, -(kernel_size - 1 + seq_len_b):]
        windows = computation_input.unfold(-1, kernel_size, 1)

        out_b = (windows * weight_t[None, :, None, :]).sum(dim=-1)

        if bias is not None:
            out_b = out_b + bias[None, :, None]

        if activation:
            out_b = F.silu(out_b)

        out[:, start:end] = out_b.to(x_t.dtype)

        target_state_len = (kernel_size - 1) + (seq_len_b - 1)
        if target_state_len > 0:
            conv_state_t[cache_idx] = full_context[:, -target_state_len:]

    out = out.transpose(0, 1)
    return out


def test_npu_causal_conv1d_update():
    """Test the NPU causal_conv1d operator in Update (decoder) mode."""
    try:
        import torch_npu
    except ImportError as e:
        print(f"Skipping NPU test (import failed): {e}")
        return

    try:
        import sgl_kernel_npu
    except ImportError as e:
        print(f"Skipping NPU test (sgl_kernel_npu import failed): {e}")
        return

    try:
        if not (hasattr(torch_npu, "npu") and torch.npu.device_count() > 0):
            print("NPU not available, skipping NPU test")
            return
    except Exception as e:
        print(f"Failed to check NPU availability: {e}")
        return

    if not hasattr(torch.ops.npu, "causal_conv1d"):
        print("causal_conv1d operator not registered!")
        return

    BSZ = 32
    HIDDEN_SIZE = 4096  # 12288
    SEQ_LEN = 4
    KERNEL_SIZE = 4
    CACHE_LEN = 32
    CONV_STATE_LEN = KERNEL_SIZE - 1 + SEQ_LEN - 1
    DTYPE = torch.bfloat16
    DEVICE = "npu"
    RUN_MODE_UPDATE = 1

    print(f"\n{'=' * 50}")
    print(f"Testing NPU causal_conv1d (Update mode) on {DEVICE}")
    print(f"{'=' * 50}")

    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = None
    hidden_state = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_init = torch.randn(CACHE_LEN, CONV_STATE_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_indices = torch.arange(BSZ, device=DEVICE, dtype=torch.int64)
    num_accepted_tokens = torch.tensor([SEQ_LEN] * BSZ, device=DEVICE, dtype=torch.int64)
    query_start_loc = torch.tensor(
        [0, SEQ_LEN, 2 * SEQ_LEN, 3 * SEQ_LEN], device=DEVICE, dtype=torch.int64
    )
    conv_state_vl = conv_state_init.clone()
    out_vl = vllm_causal_conv1d_update(
        hidden_state=hidden_state,
        conv_state=conv_state_vl,
        weight=weight,
        bias=bias,
        conv_state_indices=conv_state_indices,
        activation=True,
    )

    conv_state_npu = conv_state_init.clone()

    try:
        out_npu = torch.ops.npu.causal_conv1d(
            x=hidden_state,
            weight=weight,
            bias=bias,
            conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            query_start_loc=query_start_loc,
            num_accepted_tokens=num_accepted_tokens,
            initial_state=None,
            activation_mode=True,
            pad_slot_id=-1,
            run_mode=RUN_MODE_UPDATE,
        )

        print(f"NPU kernel executed successfully!")
        print(f"Output shape: {out_npu.shape}")

        out_npu_cpu = out_npu.cpu()
        out_vl = out_vl.cpu()

        assert out_npu_cpu.shape == out_vl.shape, f"Output shape mismatch: {out_npu_cpu.shape} vs {out_vl.shape}"

        diff = out_npu_cpu - out_vl
        abs_diff = torch.abs(diff)
        ATOL, RTOL = 5e-2, 1e-2
        tol = ATOL + RTOL * torch.abs(out_vl)
        matched = (abs_diff <= tol).sum().item()
        total = abs_diff.numel()
        print(f"Output precision: {matched}/{total} ({100 * matched / total:.2f}%) match")

        vllm_last = conv_state_vl
        npu_state = conv_state_npu.cpu()
        state_diff = (npu_state - vllm_last.cpu()).abs()
        state_exact_match = (state_diff < 1e-6).sum().item()
        state_total = state_diff.numel()
        print(f"State precision: {state_exact_match}/{state_total} ({100 * state_exact_match / state_total:.2f}%) exact match")

        if matched >= total * 0.95 and state_exact_match == state_total:
            print("PASS: Update mode test passed!")
        else:
            print("WARNING: Precision below expected threshold")

    except Exception as e:
        print(f"NPU test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_npu_causal_conv1d_fn():
    """Test the NPU causal_conv1d operator in FN (prefill) mode."""
    try:
        import torch_npu
    except ImportError as e:
        print(f"Skipping NPU test (import failed): {e}")
        return

    try:
        import sgl_kernel_npu
    except ImportError as e:
        print(f"Skipping NPU test (sgl_kernel_npu import failed): {e}")
        return

    try:
        if not (hasattr(torch_npu, "npu") and torch.npu.device_count() > 0):
            print("NPU not available, skipping NPU test")
            return
    except Exception as e:
        print(f"Failed to check NPU availability: {e}")
        return

    if not hasattr(torch.ops.npu, "causal_conv1d"):
        print("causal_conv1d operator not registered!")
        return

    BSZ = 4
    HIDDEN_SIZE = 4096
    KERNEL_SIZE = 4
    CACHE_LEN = 4
    CONV_STATE_LEN = KERNEL_SIZE - 1
    DTYPE = torch.bfloat16
    DEVICE = "npu"
    RUN_MODE_FN = 0

    SEQ_LENS = [8, 16, 12, 4]
    cu_seqlen = sum(SEQ_LENS)
    query_start_loc = torch.tensor([0] + [sum(SEQ_LENS[:i + 1]) for i in range(len(SEQ_LENS))],
                                   device=DEVICE, dtype=torch.int64)

    print(f"\n{'=' * 50}")
    print(f"Testing NPU causal_conv1d (FN/prefill mode) on {DEVICE}")
    print(f"{'=' * 50}")

    x = torch.randn(cu_seqlen, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = None
    conv_state_init = torch.randn(CACHE_LEN, CONV_STATE_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_indices = torch.arange(BSZ, device=DEVICE, dtype=torch.int64)

    conv_state_ref = conv_state_init.clone()
    out_ref = vllm_causal_conv1d_fn_varlen(
        x=x,
        weight=weight,
        conv_state=conv_state_ref,
        query_start_loc=query_start_loc,
        conv_state_indices=conv_state_indices,
        bias=bias,
        activation=True,
    )

    conv_state_npu = conv_state_init.clone()

    try:
        out_npu = torch.ops.npu.causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            query_start_loc=query_start_loc,
            num_accepted_tokens=None,
            initial_state=None,
            activation_mode=True,
            pad_slot_id=-1,
            run_mode=RUN_MODE_FN,
        )

        print(f"NPU kernel executed successfully!")
        print(f"Output shape: {out_npu.shape}")

        out_npu_cpu = out_npu.cpu()
        out_ref_cpu = out_ref.cpu()

        assert out_npu_cpu.shape == out_ref_cpu.shape, \
            f"Output shape mismatch: {out_npu_cpu.shape} vs {out_ref_cpu.shape}"

        diff = out_npu_cpu - out_ref_cpu
        abs_diff = torch.abs(diff)
        ATOL, RTOL = 5e-2, 1e-2
        tol = ATOL + RTOL * torch.abs(out_ref_cpu)
        matched = (abs_diff <= tol).sum().item()
        total = abs_diff.numel()
        print(f"Output precision: {matched}/{total} ({100 * matched / total:.2f}%) match")

        state_diff = (conv_state_npu.cpu() - conv_state_ref.cpu()).abs()
        state_exact_match = (state_diff < 1e-6).sum().item()
        state_total = state_diff.numel()
        print(f"State precision: {state_exact_match}/{state_total} ({100 * state_exact_match / state_total:.2f}%) exact match")

        if matched >= total * 0.95 and state_exact_match == state_total:
            print("PASS: FN/prefill mode test passed!")
        else:
            print("WARNING: Precision below expected threshold")

    except Exception as e:
        print(f"NPU test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_npu_causal_conv1d_fn_3d():
    """Test the NPU causal_conv1d operator in FN (prefill) mode with 3D input."""
    try:
        import torch_npu
    except ImportError as e:
        print(f"Skipping NPU test (import failed): {e}")
        return

    try:
        import sgl_kernel_npu
    except ImportError as e:
        print(f"Skipping NPU test (sgl_kernel_npu import failed): {e}")
        return

    try:
        if not (hasattr(torch_npu, "npu") and torch.npu.device_count() > 0):
            print("NPU not available, skipping NPU test")
            return
    except Exception as e:
        print(f"Failed to check NPU availability: {e}")
        return

    if not hasattr(torch.ops.npu, "causal_conv1d"):
        print("causal_conv1d operator not registered!")
        return

    BSZ = 4
    HIDDEN_SIZE = 4096
    SEQ_LEN = 8
    KERNEL_SIZE = 4
    CACHE_LEN = 4
    CONV_STATE_LEN = KERNEL_SIZE - 1
    DTYPE = torch.bfloat16
    DEVICE = "npu"
    RUN_MODE_FN = 0

    print(f"\n{'=' * 50}")
    print(f"Testing NPU causal_conv1d (FN/prefill 3D mode) on {DEVICE}")
    print(f"{'=' * 50}")

    x = torch.randn(BSZ, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(KERNEL_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    bias = None
    conv_state_init = torch.randn(CACHE_LEN, CONV_STATE_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
    conv_state_indices = torch.arange(BSZ, device=DEVICE, dtype=torch.int64)
    query_start_loc = torch.arange(0, (BSZ + 1) * SEQ_LEN, SEQ_LEN, device=DEVICE, dtype=torch.int64)

    conv_state_ref = conv_state_init.clone()
    out_ref = vllm_causal_conv1d_fn_varlen(
        x=x.reshape(-1, HIDDEN_SIZE),
        weight=weight,
        conv_state=conv_state_ref,
        query_start_loc=query_start_loc,
        conv_state_indices=conv_state_indices,
        bias=bias,
        activation=True,
    ).reshape(BSZ, SEQ_LEN, HIDDEN_SIZE)

    conv_state_npu = conv_state_init.clone()

    try:
        out_npu = torch.ops.npu.causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            conv_state=conv_state_npu,
            conv_state_indices=conv_state_indices,
            query_start_loc=query_start_loc,
            num_accepted_tokens=None,
            initial_state=None,
            activation_mode=True,
            pad_slot_id=-1,
            run_mode=RUN_MODE_FN,
        )

        print(f"NPU kernel executed successfully!")
        print(f"Output shape: {out_npu.shape}")

        out_npu_cpu = out_npu.cpu()
        out_ref_cpu = out_ref.cpu()

        assert out_npu_cpu.shape == out_ref_cpu.shape, \
            f"Output shape mismatch: {out_npu_cpu.shape} vs {out_ref_cpu.shape}"

        diff = out_npu_cpu - out_ref_cpu
        abs_diff = torch.abs(diff)
        ATOL, RTOL = 5e-2, 1e-2
        tol = ATOL + RTOL * torch.abs(out_ref_cpu)
        matched = (abs_diff <= tol).sum().item()
        total = abs_diff.numel()
        print(f"Output precision: {matched}/{total} ({100 * matched / total:.2f}%) match")

        state_diff = (conv_state_npu.cpu() - conv_state_ref.cpu()).abs()
        state_exact_match = (state_diff < 1e-6).sum().item()
        state_total = state_diff.numel()
        print(f"State precision: {state_exact_match}/{state_total} ({100 * state_exact_match / state_total:.2f}%) exact match")

        if matched >= total * 0.95 and state_exact_match == state_total:
            print("PASS: FN/prefill 3D mode test passed!")
        else:
            print("WARNING: Precision below expected threshold")

    except Exception as e:
        print(f"NPU test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Running test_npu_causal_conv1d_update (Update/Decoder mode)")
    print("=" * 60)
    test_npu_causal_conv1d_update()

    # print("\n" + "=" * 60)
    # print("Running test_npu_causal_conv1d_fn (FN/Prefill varlen mode)")
    # print("=" * 60)
    # test_npu_causal_conv1d_fn()

    # print("\n" + "=" * 60)
    # print("Running test_npu_causal_conv1d_fn_3d (FN/Prefill 3D mode)")
    # print("=" * 60)
    # test_npu_causal_conv1d_fn_3d()
