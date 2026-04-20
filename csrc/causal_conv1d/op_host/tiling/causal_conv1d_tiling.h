/*!
 * \file causal_conv1d_tiling.h
 * \brief host-side tiling helpers for causal_conv1d
 */

#ifndef CAUSAL_CONV1D_TILING_HOST_H_
#define CAUSAL_CONV1D_TILING_HOST_H_

#include <array>
#include <cstdint>
#include <limits>

#include "causal_conv1d_tiling_data.h"

namespace SGLang {
namespace CausalConv1d {

enum FnExecutionPlan : int64_t {
    FN_EXECUTION_PLAN_INVALID = 0,
    FN_EXECUTION_PLAN_CUTBS = 1,
    FN_EXECUTION_PLAN_CUTBSD = 2,
};

inline constexpr int64_t ResolveFnExecutionPlan(int64_t baseDimCnt)
{
    if (baseDimCnt <= 0) {
        return FN_EXECUTION_PLAN_INVALID;
    }
    if (baseDimCnt <= 1) {
        return FN_EXECUTION_PLAN_CUTBS;
    }
    return FN_EXECUTION_PLAN_CUTBSD;
}

struct DimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

struct VarlenTokenTileChoice {
    bool enabled = false;
    int64_t tokenBlockSize = 0;
    int64_t tokenBlockCnt = 0;
    int64_t gridSize = 0;
};

struct TokenCoreMappingChoice {
    int64_t tokenCoreBudget = 0;
    int64_t tokenBlocksPerCore = 0;
    int64_t tokenCoreTailCnt = 0;
    int64_t blockDim = 0;
};

constexpr int64_t MAX_FN_TOKEN_SEQ_RANGE_COUNT = 128;

struct FnTokenSeqRangePlan {
    bool enabled = false;
    int64_t rangeCount = 0;
    int64_t tokenTileStartSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
    int64_t tokenTileEndSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
};

enum FnTilingCaseKind : int64_t {
    FN_TILING_CASE_INVALID = 0,
    FN_TILING_CASE_TOKEN_FIRST = 1,
    FN_TILING_CASE_TOKEN_DIM_CO_SPLIT = 2,
};

struct FnHostPlan {
    FnTilingCaseKind caseKind = FN_TILING_CASE_INVALID;
    FnExecutionPlan executionPlan = FN_EXECUTION_PLAN_INVALID;
    DimTileChoice baseDimChoice;
    VarlenTokenTileChoice tokenBlockChoice;
    TokenCoreMappingChoice tokenCoreMapping;
    FnTokenSeqRangePlan tokenSeqRangePlan;
};

inline int64_t CeilDivInt64(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t AlignDownInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return (value / align) * align;
}

inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return CeilDivInt64(value, align) * align;
}

constexpr int64_t DIM_ALIGN_BYTES = 32;
constexpr int64_t BF16_FP16_ELEM_BYTES = 2;
constexpr int64_t DIM_ALIGN_ELEMS = DIM_ALIGN_BYTES / BF16_FP16_ELEM_BYTES;
constexpr int64_t MAX_DIM_TILE_SIZE = 4096;
constexpr int64_t FN_UB_RESERVED_BYTES = 512;
constexpr int64_t RING_SLOT_CNT = 5;
constexpr int64_t FN_OUT_SLOT_CNT = 2;
constexpr int64_t FN_CALC_FP32_SLOT_CNT = 8;

inline DimTileChoice ChooseCanonicalUpdateBaseDimChoice(int64_t batch, int64_t dim, uint32_t coreNum)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};

    auto chooseOnce = [&](bool requireExactDiv) -> DimTileChoice {
        DimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        DimTileChoice bestUnder;

        for (int64_t baseDim : candidates) {
            if (baseDim <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % baseDim != 0)) {
                continue;
            }

            const int64_t baseDimCnt = requireExactDiv ? (dim / baseDim) : CeilDivInt64(dim, baseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }

            if (gridSize >= static_cast<int64_t>(coreNum)) {
                const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
                if (gap < bestOverGap) {
                    bestOver = {baseDim, baseDimCnt, gridSize};
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && baseDim < bestUnder.baseDim)) {
                bestUnder = {baseDim, baseDimCnt, gridSize};
            }
        }

        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    DimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    return result;
}

inline int64_t ComputeFnUbLimitedBaseDim(uint64_t ubSize)
{
    if (ubSize <= static_cast<uint64_t>(FN_UB_RESERVED_BYTES)) {
        return 0;
    }

    const int64_t bytesPerElem = (RING_SLOT_CNT * BF16_FP16_ELEM_BYTES) + (FN_OUT_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                 (FN_CALC_FP32_SLOT_CNT * static_cast<int64_t>(sizeof(float)));
    const int64_t budgetBytes = static_cast<int64_t>(ubSize) - FN_UB_RESERVED_BYTES;
    const int64_t ubLimitedBaseDim = AlignDownInt64(budgetBytes / bytesPerElem, DIM_ALIGN_ELEMS);
    return std::min<int64_t>(MAX_DIM_TILE_SIZE, ubLimitedBaseDim);
}

inline DimTileChoice ChooseFnTokenFirstBaseDimChoice(int64_t dim)
{
    if (dim <= 0 || dim > MAX_DIM_TILE_SIZE) {
        return {};
    }
    return {dim, 1, 1};
}

inline DimTileChoice ChooseFnTokenDimCoSplitBaseDimChoice(int64_t dim, uint64_t ubSize, uint32_t coreNum)
{
    if (dim <= 0) {
        return {};
    }

    const int64_t ubLimitedBaseDim = ComputeFnUbLimitedBaseDim(ubSize);
    if (ubLimitedBaseDim <= 0) {
        return {};
    }

    DimTileChoice result;
    result.baseDim = ubLimitedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;

    if (coreNum == 0 || result.baseDimCnt <= 1 || result.baseDimCnt >= static_cast<int64_t>(coreNum) ||
        (coreNum % result.baseDimCnt == 0)) {
        return result;
    }

    int64_t adjustedBaseDimCnt = result.baseDimCnt;
    while (adjustedBaseDimCnt < static_cast<int64_t>(coreNum) && (coreNum % adjustedBaseDimCnt != 0)) {
        ++adjustedBaseDimCnt;
    }

    if (adjustedBaseDimCnt >= static_cast<int64_t>(coreNum)) {
        return result;
    }

    const int64_t adjustedBaseDim = AlignUpInt64(CeilDivInt64(dim, adjustedBaseDimCnt), DIM_ALIGN_ELEMS);
    if (adjustedBaseDim <= 0 || adjustedBaseDim > ubLimitedBaseDim || adjustedBaseDim > MAX_DIM_TILE_SIZE) {
        return result;
    }

    result.baseDim = adjustedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;
    return result;
}

inline int64_t ResolveFnTokenCoreBudget(int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    if (baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return 0;
    }

    int64_t tokenCoreBudget = static_cast<int64_t>(coreNum);
    if (fnExecutionPlan == FN_EXECUTION_PLAN_CUTBSD) {
        tokenCoreBudget = std::max<int64_t>(1, tokenCoreBudget / baseDimCnt);
    }
    return tokenCoreBudget;
}

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                      FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    const int64_t tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (cuSeqlen <= 0 || tokenCoreBudget <= 0) {
        return tokenBlockChoice;
    }

    tokenBlockChoice.enabled = true;
    const int64_t idealBlockSize = CeilDivInt64(cuSeqlen, tokenCoreBudget);
    tokenBlockChoice.tokenBlockSize = (idealBlockSize > 0) ? idealBlockSize : 1;
    tokenBlockChoice.tokenBlockCnt = CeilDivInt64(cuSeqlen, tokenBlockChoice.tokenBlockSize);
    tokenBlockChoice.gridSize = tokenBlockChoice.tokenBlockCnt * baseDimCnt;
    return tokenBlockChoice;
}

inline TokenCoreMappingChoice BuildFnTokenCoreMappingChoice(int64_t tokenBlockCnt, int64_t baseDimCnt,
                                                            FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    TokenCoreMappingChoice mapping;
    mapping.tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (tokenBlockCnt <= 0 || mapping.tokenCoreBudget <= 0 || baseDimCnt <= 0) {
        return mapping;
    }

    mapping.tokenBlocksPerCore = CeilDivInt64(tokenBlockCnt, mapping.tokenCoreBudget);
    mapping.tokenCoreTailCnt =
        tokenBlockCnt - (std::max<int64_t>(0, mapping.tokenBlocksPerCore - 1) * mapping.tokenCoreBudget);
    if (mapping.tokenCoreTailCnt <= 0) {
        mapping.tokenCoreTailCnt = mapping.tokenCoreBudget;
    }
    mapping.blockDim = mapping.tokenCoreBudget * baseDimCnt;
    return mapping;
}

inline FnTokenSeqRangePlan BuildFnTokenSeqRangePlan(const int64_t *qslData, int64_t batch, int64_t tokenBlockSize,
                                                    int64_t tokenBlockCnt)
{
    FnTokenSeqRangePlan plan;
    if (qslData == nullptr || batch <= 0 || tokenBlockSize <= 0 || tokenBlockCnt <= 0 ||
        tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
        return plan;
    }

    plan.enabled = true;
    plan.rangeCount = tokenBlockCnt;
    int64_t seq = 0;
    for (int64_t tokenTileId = 0; tokenTileId < tokenBlockCnt; ++tokenTileId) {
        const int64_t tokenStart = tokenTileId * tokenBlockSize;
        const int64_t tokenEnd = tokenStart + tokenBlockSize;

        while (seq < batch && qslData[seq + 1] <= tokenStart) {
            ++seq;
        }

        int64_t endSeq = seq;
        while (endSeq < batch && qslData[endSeq] < tokenEnd) {
            ++endSeq;
        }

        plan.tokenTileStartSeq[tokenTileId] = seq;
        plan.tokenTileEndSeq[tokenTileId] = endSeq;
    }
    return plan;
}

inline VarlenTokenTileChoice ChooseUnifiedFnTokenBlockPlan(int64_t inputMode, int64_t batch, int64_t cuSeqlen,
                                                           int64_t hasNumAcceptedTokens,
                                                           const DimTileChoice &baseDimChoice,
                                                           FnExecutionPlan fnExecutionPlan,
                                                           uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    if ((inputMode != 0 && inputMode != 1) || batch <= 0 || cuSeqlen <= 0 ||
        baseDimChoice.baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return tokenBlockChoice;
    }
    if (hasNumAcceptedTokens != 0) {
        return tokenBlockChoice;
    }

    tokenBlockChoice = ChooseFnTokenBlockChoice(cuSeqlen, baseDimChoice.baseDimCnt, fnExecutionPlan, coreNum);
    return tokenBlockChoice;
}

inline FnHostPlan ChooseFnHostPlan(int64_t inputMode, int64_t batch, int64_t cuSeqlen, int64_t dim,
                                   int64_t hasNumAcceptedTokens,
                                   uint64_t ubSize, uint32_t coreNum)
{
    FnHostPlan plan;
    if ((inputMode != 0 && inputMode != 1) || batch <= 0 || cuSeqlen <= 0 || dim <= 0 || coreNum == 0) {
        return plan;
    }

    if (dim <= MAX_DIM_TILE_SIZE) {
        plan.caseKind = FN_TILING_CASE_TOKEN_FIRST;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBS;
        plan.baseDimChoice = ChooseFnTokenFirstBaseDimChoice(dim);
    } else {
        plan.caseKind = FN_TILING_CASE_TOKEN_DIM_CO_SPLIT;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBSD;
        plan.baseDimChoice = ChooseFnTokenDimCoSplitBaseDimChoice(dim, ubSize, coreNum);
    }

    if (plan.baseDimChoice.baseDim <= 0 || plan.baseDimChoice.baseDimCnt <= 0) {
        return {};
    }

    plan.baseDimChoice.gridSize = batch * plan.baseDimChoice.baseDimCnt;
    plan.tokenBlockChoice =
        ChooseUnifiedFnTokenBlockPlan(inputMode, batch, cuSeqlen, hasNumAcceptedTokens,
                                      plan.baseDimChoice, plan.executionPlan, coreNum);
    if (!plan.tokenBlockChoice.enabled || plan.tokenBlockChoice.tokenBlockSize <= 0 ||
        plan.tokenBlockChoice.tokenBlockCnt <= 0 || plan.tokenBlockChoice.gridSize <= 0) {
        return {};
    }

    plan.tokenCoreMapping = BuildFnTokenCoreMappingChoice(plan.tokenBlockChoice.tokenBlockCnt,
                                                          plan.baseDimChoice.baseDimCnt, plan.executionPlan, coreNum);
    if (plan.tokenCoreMapping.tokenCoreBudget <= 0 || plan.tokenCoreMapping.blockDim <= 0) {
        return {};
    }
    if (plan.tokenCoreMapping.blockDim > static_cast<int64_t>(coreNum)) {
        plan.tokenCoreMapping.blockDim = static_cast<int64_t>(coreNum);
    }
    return plan;
}

struct CausalConv1dTilingResult {
    sglang::npu_kernel::CausalConv1dTilingData tilingData;
    int64_t effectiveGridSize = 0;
    int64_t blockDim = 0;
};

inline CausalConv1dTilingResult ComputeTilingData(int64_t batch, int64_t cu_seqlen, int64_t seq_len, int64_t input_mode, int64_t dim,
                              int64_t width, int64_t state_len, int64_t num_cache_lines, bool has_bias,
                              bool activation_mode, int64_t pad_slot_id, int64_t run_mode,
                              bool has_num_accepted_tokens, bool has_cache_indices, bool has_initial_state_mode,
                              uint64_t ubSize, uint32_t coreNum)
{
    CausalConv1dTilingResult result;
    auto &tiling_data = result.tilingData;

    tiling_data.dim = dim;
    tiling_data.cuSeqlen = cu_seqlen;
    tiling_data.seqLen = seq_len;
    tiling_data.inputMode = input_mode;
    tiling_data.width = width;
    tiling_data.stateLen = state_len;
    tiling_data.numCacheLines = num_cache_lines;
    tiling_data.batch = batch;
    tiling_data.activationMode = activation_mode ? 1 : 0;
    tiling_data.padSlotId = pad_slot_id;
    tiling_data.hasBias = has_bias ? 1 : 0;
    tiling_data.runMode = run_mode;
    tiling_data.hasNumAcceptedTokens = has_num_accepted_tokens ? 1 : 0;
    tiling_data.hasCacheIndices = has_cache_indices ? 1 : 0;
    tiling_data.hasInitialStateMode = has_initial_state_mode ? 1 : 0;

    tiling_data.tokenBlockSize = 0;
    tiling_data.tokenBlockCnt = 0;
    tiling_data.hasExplicitTokenSeqRanges = 0;
    tiling_data.explicitTokenSeqRangeCount = 0;

    const bool isFn = (run_mode == sglang::npu_kernel::CAUSAL_CONV1D_RUN_MODE_FN);

    DimTileChoice baseDimChoice;
    FnExecutionPlan fnExecutionPlan = FN_EXECUTION_PLAN_INVALID;
    FnHostPlan fnHostPlan;

    if (isFn) {
        fnHostPlan = ChooseFnHostPlan(input_mode, batch, cu_seqlen, dim,
                                      tiling_data.hasNumAcceptedTokens,
                                      ubSize, coreNum);
        baseDimChoice = fnHostPlan.baseDimChoice;
        fnExecutionPlan = fnHostPlan.executionPlan;
    } else {
        baseDimChoice = ChooseCanonicalUpdateBaseDimChoice(batch, dim, coreNum);
    }

    int64_t effectiveGridSize = baseDimChoice.gridSize;

    if (isFn && fnHostPlan.caseKind != FN_TILING_CASE_INVALID &&
        fnHostPlan.executionPlan != FN_EXECUTION_PLAN_INVALID &&
        fnHostPlan.tokenBlockChoice.enabled &&
        fnHostPlan.tokenBlockChoice.tokenBlockSize > 0 &&
        fnHostPlan.tokenBlockChoice.tokenBlockCnt > 0 &&
        fnHostPlan.tokenBlockChoice.gridSize > 0 &&
        fnHostPlan.tokenCoreMapping.tokenCoreBudget > 0 &&
        fnHostPlan.tokenCoreMapping.blockDim > 0) {

        tiling_data.tokenBlockSize = fnHostPlan.tokenBlockChoice.tokenBlockSize;
        tiling_data.tokenBlockCnt = fnHostPlan.tokenBlockChoice.tokenBlockCnt;
        effectiveGridSize = fnHostPlan.tokenBlockChoice.gridSize;

        const int64_t mappedBlockDim = std::min<int64_t>(effectiveGridSize, fnHostPlan.tokenCoreMapping.blockDim);
        result.blockDim = (mappedBlockDim > 0) ? mappedBlockDim : effectiveGridSize;
    } else if (isFn) {
        result.blockDim = effectiveGridSize;
    } else {
        result.blockDim = (effectiveGridSize < static_cast<int64_t>(coreNum)) ? effectiveGridSize : static_cast<int64_t>(coreNum);
    }

    if (result.blockDim <= 0) {
        result.blockDim = 1;
    }

    tiling_data.baseDim = baseDimChoice.baseDim;
    tiling_data.baseDimCnt = baseDimChoice.baseDimCnt;
    result.effectiveGridSize = effectiveGridSize;
    return result;
}

inline CausalConv1dTilingResult ComputeTilingDataWithSeqRanges(int64_t batch, int64_t cu_seqlen, int64_t seq_len, int64_t input_mode, int64_t dim,
                                           int64_t width, int64_t state_len, int64_t num_cache_lines, bool has_bias,
                                           bool activation_mode, int64_t pad_slot_id, int64_t run_mode,
                                           bool has_num_accepted_tokens, bool has_cache_indices, bool has_initial_state_mode,
                                           uint64_t ubSize, uint32_t coreNum,
                                           const int64_t *qslData)
{
    CausalConv1dTilingResult result = ComputeTilingData(
        batch, cu_seqlen, seq_len, input_mode, dim, width, state_len, num_cache_lines,
        has_bias, activation_mode, pad_slot_id, run_mode,
        has_num_accepted_tokens, has_cache_indices, has_initial_state_mode,
        ubSize, coreNum);

    auto &tiling_data = result.tilingData;
    const bool isFn = (run_mode == sglang::npu_kernel::CAUSAL_CONV1D_RUN_MODE_FN);
    if (isFn && input_mode == 0 && tiling_data.tokenBlockCnt > 0 && qslData != nullptr) {
        FnTokenSeqRangePlan seqRangePlan =
            BuildFnTokenSeqRangePlan(qslData, batch, tiling_data.tokenBlockSize, tiling_data.tokenBlockCnt);
        if (seqRangePlan.enabled) {
            tiling_data.hasExplicitTokenSeqRanges = 1;
            tiling_data.explicitTokenSeqRangeCount = seqRangePlan.rangeCount;
            for (int64_t i = 0; i < seqRangePlan.rangeCount; ++i) {
                tiling_data.tokenTileStartSeq[i] = seqRangePlan.tokenTileStartSeq[i];
                tiling_data.tokenTileEndSeq[i] = seqRangePlan.tokenTileEndSeq[i];
            }
        }
    }
    return result;
}

}  // namespace CausalConv1d
}  // namespace SGLang

#endif  // CAUSAL_CONV1D_TILING_HOST_H_
