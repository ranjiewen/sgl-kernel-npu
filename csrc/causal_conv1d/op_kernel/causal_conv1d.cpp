#include "causal_conv1d_fn.h"
#include "causal_conv1d_update.h"

namespace {

using sglang::npu_kernel::CausalConv1dTilingData;

__aicore__ inline void InitTilingData(GM_ADDR tiling, CausalConv1dTilingData &tiling_data)
{
    auto gm_tiling = reinterpret_cast<__gm__ CausalConv1dTilingData *>(tiling);
    tiling_data.batch = gm_tiling->batch;
    tiling_data.seqLen = gm_tiling->seqLen;
    tiling_data.dim = gm_tiling->dim;
    tiling_data.width = gm_tiling->width;
    tiling_data.stateLen = gm_tiling->stateLen;
    tiling_data.numCacheLines = gm_tiling->numCacheLines;
    tiling_data.cuSeqlen = gm_tiling->cuSeqlen;
    tiling_data.inputMode = gm_tiling->inputMode;
    tiling_data.hasBias = gm_tiling->hasBias;
    tiling_data.activationMode = gm_tiling->activationMode;
    tiling_data.padSlotId = gm_tiling->padSlotId;
    tiling_data.baseDim = gm_tiling->baseDim;
    tiling_data.baseDimCnt = gm_tiling->baseDimCnt;
    tiling_data.runMode = gm_tiling->runMode;
    tiling_data.hasNumAcceptedTokens = gm_tiling->hasNumAcceptedTokens;
    tiling_data.hasCacheIndices = gm_tiling->hasCacheIndices;
    tiling_data.hasInitialStateMode = gm_tiling->hasInitialStateMode;
    tiling_data.tokenBlockSize = gm_tiling->tokenBlockSize;
    tiling_data.tokenBlockCnt = gm_tiling->tokenBlockCnt;
    tiling_data.hasExplicitTokenSeqRanges = gm_tiling->hasExplicitTokenSeqRanges;
    tiling_data.explicitTokenSeqRangeCount = gm_tiling->explicitTokenSeqRangeCount;
    for (int64_t i = 0; i < sglang::npu_kernel::CAUSAL_CONV1D_MAX_TOKEN_SEQ_RANGE_COUNT; ++i) {
        tiling_data.tokenTileStartSeq[i] = gm_tiling->tokenTileStartSeq[i];
        tiling_data.tokenTileEndSeq[i] = gm_tiling->tokenTileEndSeq[i];
    }
}

}  // namespace

extern "C" __global__ __aicore__ void causal_conv1d_bfloat16_t(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                                       GM_ADDR convStates, GM_ADDR convStateIndices,
                                                                       GM_ADDR queryStartLoc, GM_ADDR numAcceptedTokens,
                                                                       GM_ADDR initialState,
                                                                       GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CausalConv1dTilingData tiling_data;
    InitTilingData(tiling, tiling_data);

    if (tiling_data.runMode == sglang::npu_kernel::CAUSAL_CONV1D_RUN_MODE_UPDATE) {
        NsCausalConv1d::RunCausalConv1dUpdate<bfloat16_t>(
            x, weight, bias, convStates, convStateIndices, queryStartLoc, numAcceptedTokens, initialState,
            y, workspace, &tiling_data);
    } else {
        NsCausalConv1d::RunCausalConv1dFn<bfloat16_t>(
            x, weight, bias, convStates, convStateIndices, queryStartLoc, numAcceptedTokens, initialState,
            y, workspace, &tiling_data);
    }
}

extern "C" __global__ __aicore__ void causal_conv1d_half(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                               GM_ADDR convStates, GM_ADDR convStateIndices,
                                                               GM_ADDR queryStartLoc, GM_ADDR numAcceptedTokens,
                                                               GM_ADDR initialState,
                                                               GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CausalConv1dTilingData tiling_data;
    InitTilingData(tiling, tiling_data);

    if (tiling_data.runMode == sglang::npu_kernel::CAUSAL_CONV1D_RUN_MODE_UPDATE) {
        NsCausalConv1d::RunCausalConv1dUpdate<half>(
            x, weight, bias, convStates, convStateIndices, queryStartLoc, numAcceptedTokens, initialState,
            y, workspace, &tiling_data);
    } else {
        NsCausalConv1d::RunCausalConv1dFn<half>(
            x, weight, bias, convStates, convStateIndices, queryStartLoc, numAcceptedTokens, initialState,
            y, workspace, &tiling_data);
    }
}
