/*!
 * \file causal_conv1d_tiling_data.h
 * \brief tiling data struct for causal_conv1d
 */

#ifndef CAUSAL_CONV1D_TILING_DATA_H_
#define CAUSAL_CONV1D_TILING_DATA_H_

#include <cstdint>

namespace sglang {
namespace npu_kernel {

constexpr int64_t CAUSAL_CONV1D_RUN_MODE_FN = 0;
constexpr int64_t CAUSAL_CONV1D_RUN_MODE_UPDATE = 1;

constexpr int64_t CAUSAL_CONV1D_MAX_TOKEN_SEQ_RANGE_COUNT = 128;

struct CausalConv1dTilingData {
    int64_t dim = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t inputMode = 0;

    int64_t width = 0;
    int64_t stateLen = 0;
    int64_t numCacheLines = 0;

    int64_t batch = 0;

    int64_t activationMode = 0;
    int64_t padSlotId = 0;
    int64_t hasBias = 0;

    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;

    int64_t runMode = 0;

    int64_t hasNumAcceptedTokens = 0;
    int64_t hasCacheIndices = 0;
    int64_t hasInitialStateMode = 0;

    int64_t tokenBlockSize = 0;
    int64_t tokenBlockCnt = 0;

    int64_t hasExplicitTokenSeqRanges = 0;
    int64_t explicitTokenSeqRangeCount = 0;
    int64_t tokenTileStartSeq[CAUSAL_CONV1D_MAX_TOKEN_SEQ_RANGE_COUNT] = {};
    int64_t tokenTileEndSeq[CAUSAL_CONV1D_MAX_TOKEN_SEQ_RANGE_COUNT] = {};
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_TILING_DATA_H_
