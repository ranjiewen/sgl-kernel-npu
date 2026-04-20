#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <functional>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/causal_conv1d_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "common_tiling.h"
#include "common.h"
#include "stub/aclrtlaunch_causal_conv1d_bfloat16_t.h"
#include "stub/aclrtlaunch_causal_conv1d_half.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;

uint32_t conv1dCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> conv1dCaptureMap;

struct CausalConv1dTilingKey {
    int64_t batch;
    int64_t seqLen;
    int64_t dim;
    int64_t width;
    int64_t stateLen;
    int64_t hasIndices;
    int64_t hasBias;
    int64_t hasNumAccept;
    int64_t hasQueryLoc;
    int64_t hasInitState;
    int64_t activationMode;
    int64_t padSlotId;
    int64_t runMode;

    bool operator==(const CausalConv1dTilingKey &other) const
    {
        return batch == other.batch && seqLen == other.seqLen && dim == other.dim && width == other.width &&
               stateLen == other.stateLen && hasIndices == other.hasIndices && hasBias == other.hasBias &&
               hasNumAccept == other.hasNumAccept && hasQueryLoc == other.hasQueryLoc &&
               hasInitState == other.hasInitState && activationMode == other.activationMode &&
               padSlotId == other.padSlotId && runMode == other.runMode;
    }
};

struct CausalConv1dTilingKeyHash {
    std::size_t operator()(const CausalConv1dTilingKey &k) const
    {
        std::size_t h1 = std::hash<int64_t>{}(k.batch);
        std::size_t h2 = std::hash<int64_t>{}(k.seqLen);
        std::size_t h3 = std::hash<int64_t>{}(k.dim);
        std::size_t h4 = std::hash<int64_t>{}(k.width);
        std::size_t h5 = std::hash<int64_t>{}(k.stateLen);
        std::size_t h6 = std::hash<int64_t>{}(k.hasIndices);
        std::size_t h7 = std::hash<int64_t>{}(k.hasBias);
        std::size_t h8 = std::hash<int64_t>{}(k.hasNumAccept);
        std::size_t h9 = std::hash<int64_t>{}(k.hasQueryLoc);
        std::size_t h10 = std::hash<int64_t>{}(k.hasInitState);
        std::size_t h11 = std::hash<int64_t>{}(k.activationMode);
        std::size_t h12 = std::hash<int64_t>{}(k.padSlotId);
        std::size_t h13 = std::hash<int64_t>{}(k.runMode);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6) ^ (h8 << 7) ^ (h9 << 8) ^
               (h10 << 9) ^ (h11 << 10) ^ (h12 << 11) ^ (h13 << 12);
    }
};

HOST_API at::Tensor causal_conv1d_impl(const at::Tensor &x, const at::Tensor &weight,
                                       const at::Tensor &bias, const at::Tensor &conv_state,
                                       const at::Tensor &conv_state_indices, const at::Tensor &query_start_loc,
                                       const at::Tensor &num_accepted_tokens, const at::Tensor &initial_state,
                                       bool activation_mode, int64_t pad_slot_id, int64_t run_mode)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D [cu_seqlen, dim] or 3D [batch, seq_len, dim], got shape ",
                x.sizes());
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D tensor [width, dim], got shape ", weight.sizes());
    TORCH_CHECK(conv_state.dim() == 3, "conv_state must be 3D tensor [cache_len, state_len, dim], got shape ",
                conv_state.sizes());

    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "Only BF16 and FP16 are supported, got ", dtype);
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_state.scalar_type() == dtype, "conv_state dtype must match x dtype");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous before entering the NPU kernel.");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous.");
    TORCH_CHECK(conv_state.is_contiguous(), "conv_state must be contiguous.");

    const bool is_update_mode = (run_mode == CAUSAL_CONV1D_RUN_MODE_UPDATE);
    const bool is_fn_mode = (run_mode == CAUSAL_CONV1D_RUN_MODE_FN);

    int64_t batch = 0;
    int64_t seq_len = 0;
    int64_t dim = 0;
    int64_t cu_seqlen = 0;
    int64_t input_mode = 0;

    if (x.dim() == 2) {
        if (is_update_mode) {
            input_mode = 2;
            batch = x.size(0);
            dim = x.size(1);
            seq_len = 1;
            cu_seqlen = batch;
        } else {
            input_mode = 0;
            cu_seqlen = x.size(0);
            dim = x.size(1);
            TORCH_CHECK(query_start_loc.numel() > 0, "query_start_loc is required for 2D input (varlen mode)");
            batch = query_start_loc.size(0) - 1; // check ?
        }
    } else {
        input_mode = 1;
        batch = x.size(0);
        seq_len = x.size(1);
        dim = x.size(2);
        cu_seqlen = batch * seq_len;
    }

    const int64_t width = weight.size(0);
    TORCH_CHECK(width >= 2 && width <= 4, "width must be in [2, 4], got ", width);

    const int64_t state_len = conv_state.size(1);
    const int64_t num_cache_lines = conv_state.size(0);

    const bool has_indices = conv_state_indices.numel() > 0;
    const bool has_bias = bias.numel() > 0;
    const bool has_num_accept = num_accepted_tokens.numel() > 0;
    const bool has_query_loc = query_start_loc.numel() > 0;
    const bool has_initial_state = initial_state.numel() > 0;

    at::Tensor y = at::empty_like(x);

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendc_platform != nullptr, "Failed to acquire AscendC platform manager");

    uint64_t ubSize = 0;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int32_t max_aiv_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    int32_t workspace_size = static_cast<int32_t>(ascendc_platform->GetLibApiWorkSpaceSize());

    const int64_t *qsl_data = has_query_loc ? query_start_loc.data_ptr<int64_t>() : nullptr;
    SGLang::CausalConv1d::CausalConv1dTilingResult tiling_result =
        SGLang::CausalConv1d::ComputeTilingDataWithSeqRanges(
            batch, cu_seqlen, seq_len, input_mode, dim, width, state_len, num_cache_lines,
            has_bias, activation_mode, pad_slot_id, run_mode,
            has_num_accept, has_indices, has_initial_state,
            ubSize, static_cast<uint32_t>(max_aiv_core), qsl_data);
    CausalConv1dTilingData &tiling_data = tiling_result.tilingData;

    printf("[CausalConv1d] Tiling: runMode=%ld, inputMode=%ld, batch=%ld, dim=%ld, width=%ld, "
           "cuSeqlen=%ld, seqLen=%ld, stateLen=%ld, numCacheLines=%ld, "
           "baseDim=%ld, baseDimCnt=%ld, "
           "tokenBlockSize=%ld, tokenBlockCnt=%ld, "
           "hasBias=%ld, hasCacheIndices=%ld, hasNumAcceptedTokens=%ld, hasInitialStateMode=%ld, "
           "hasExplicitTokenSeqRanges=%ld, explicitTokenSeqRangeCount=%ld, "
           "activationMode=%ld, padSlotId=%ld, effectiveGridSize=%ld, blockDim=%ld, ubSize=%lu, coreNum=%u\n",
           tiling_data.runMode, tiling_data.inputMode, tiling_data.batch, tiling_data.dim, tiling_data.width,
           tiling_data.cuSeqlen, tiling_data.seqLen, tiling_data.stateLen, tiling_data.numCacheLines,
           tiling_data.baseDim, tiling_data.baseDimCnt,
           tiling_data.tokenBlockSize, tiling_data.tokenBlockCnt,
           tiling_data.hasBias, tiling_data.hasCacheIndices, tiling_data.hasNumAcceptedTokens, tiling_data.hasInitialStateMode,
           tiling_data.hasExplicitTokenSeqRanges, tiling_data.explicitTokenSeqRangeCount,
           tiling_data.activationMode, tiling_data.padSlotId,
           tiling_result.effectiveGridSize, tiling_result.blockDim, ubSize, static_cast<uint32_t>(max_aiv_core));

    int32_t tilingSize = (sizeof(CausalConv1dTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    at::Tensor tilingTensor;

    CausalConv1dTilingKey key{.batch = batch,
                              .seqLen = seq_len,
                              .dim = dim,
                              .width = width,
                              .stateLen = state_len,
                              .hasIndices = has_indices ? 1 : 0,
                              .hasBias = has_bias ? 1 : 0,
                              .hasNumAccept = has_num_accept ? 1 : 0,
                              .hasQueryLoc = has_query_loc ? 1 : 0,
                              .hasInitState = has_initial_state ? 1 : 0,
                              .activationMode = activation_mode ? 1 : 0,
                              .padSlotId = pad_slot_id,
                              .runMode = run_mode};
    uint64_t hashValue = CausalConv1dTilingKeyHash{}(key);

    auto copyTilingToDevice = [&]() {
        auto cpuTiling = at::empty({tilingSize}, at::kByte);
        std::memcpy(cpuTiling.data_ptr(), &tiling_data, sizeof(CausalConv1dTilingData));
        return TorchNpuHelper::CopyTensorHostToDevice(cpuTiling);
    };

    static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    if (conv1dCaptureMap.find(hashValue) != conv1dCaptureMap.end()) {
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * conv1dCaptureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (conv1dCaptureNum >= MAX_CAPTURE_NUM) {
        tilingTensor = copyTilingToDevice();
    } else {
        conv1dCaptureMap[hashValue] = conv1dCaptureNum;
        auto deviceTiling = copyTilingToDevice();
        globalTilingBuffer.slice(0, conv1dCaptureNum * tilingSize, conv1dCaptureNum * tilingSize + tilingSize)
            .copy_(deviceTiling);
        conv1dCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * conv1dCaptureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    int32_t block_dim = static_cast<int32_t>(tiling_result.blockDim);
    if (block_dim <= 0) {
        block_dim = 1;
    }

    printf("[CausalConv1d] Dispatch: block_dim=%d, dtype=%d, has_bias=%d, has_indices=%d, "
           "has_query_loc=%d, has_num_accept=%d, has_initial_state=%d\n",
           block_dim, static_cast<int>(dtype), has_bias, has_indices,
           has_query_loc, has_num_accept, has_initial_state);

    at::Tensor empty_bias = at::empty(0, x.options());
    at::Tensor empty_indices = at::empty(0, at::kLong);
    at::Tensor empty_query_loc = at::empty(0, at::kLong);
    at::Tensor empty_num_accept = at::empty(0, at::kLong);
    at::Tensor empty_initial_state = at::empty(0, at::kLong);

    if (dtype == at::kBFloat16) {
        EXEC_KERNEL_CMD(causal_conv1d_bfloat16_t, block_dim, x, weight,
                        has_bias ? bias : empty_bias,
                        conv_state,
                        has_indices ? conv_state_indices : empty_indices,
                        has_query_loc ? query_start_loc : empty_query_loc,
                        has_num_accept ? num_accepted_tokens : empty_num_accept,
                        has_initial_state ? initial_state : empty_initial_state,
                        y, workspace_tensor, tilingTensor);
    } else {
        EXEC_KERNEL_CMD(causal_conv1d_half, block_dim, x, weight,
                        has_bias ? bias : empty_bias,
                        conv_state,
                        has_indices ? conv_state_indices : empty_indices,
                        has_query_loc ? query_start_loc : empty_query_loc,
                        has_num_accept ? num_accepted_tokens : empty_num_accept,
                        has_initial_state ? initial_state : empty_initial_state,
                        y, workspace_tensor, tilingTensor);
    }

    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
