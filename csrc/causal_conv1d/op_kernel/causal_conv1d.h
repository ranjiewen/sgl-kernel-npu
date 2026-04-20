#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_common.h"

namespace NsCausalConv1d {

using namespace AscendC;
using namespace NsCausalConv1dCommon;
using sglang::npu_kernel::CausalConv1dTilingData;
using sglang::npu_kernel::CAUSAL_CONV1D_RUN_MODE_UPDATE;
using sglang::npu_kernel::CAUSAL_CONV1D_RUN_MODE_FN;

enum SeqTaskWindowMode : int32_t {
    SEQ_TASK_WINDOW_MODE_VARLEN = 0,
    SEQ_TASK_WINDOW_MODE_BATCH = 1,
    SEQ_TASK_WINDOW_MODE_DECODE2D = 2,
};

struct SeqTaskWindow {
    bool valid = false;
    int32_t start = 0;
    int32_t len = 0;
};

__aicore__ inline int32_t GetSeqTaskWindowMode(int32_t inputMode)
{
    if (inputMode == 0) {
        return SEQ_TASK_WINDOW_MODE_VARLEN;
    }
    if (inputMode == 2) {
        return SEQ_TASK_WINDOW_MODE_DECODE2D;
    }
    return SEQ_TASK_WINDOW_MODE_BATCH;
}

__aicore__ inline SeqTaskWindow BuildSeqTaskWindowVarlen(int32_t startVal, int32_t endVal)
{
    SeqTaskWindow window;
    window.start = startVal;
    window.len = endVal - startVal;
    window.valid = (window.len > 0);
    return window;
}

__aicore__ inline SeqTaskWindow BuildSeqTaskWindowBatch(int32_t seq, int32_t seqLen)
{
    SeqTaskWindow window;
    window.start = seq * seqLen;
    window.len = seqLen;
    window.valid = (window.len > 0);
    return window;
}

__aicore__ inline SeqTaskWindow BuildSeqTaskWindowDecode2D(int32_t seq)
{
    SeqTaskWindow window;
    window.valid = true;
    window.start = seq;
    window.len = 1;
    return window;
}

template <typename T>
class CausalConv1d {
public:
    __aicore__ inline CausalConv1d() = default;

protected:
    __aicore__ inline void ResetRuntimeState(const CausalConv1dTilingData *tilingData);
    __aicore__ inline void InitSharedBuffersAndEvents();
    __aicore__ inline void LoadWeightAndBias(int32_t channelStart, int32_t baseDim);
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start,
                                    int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart, int32_t baseDim,
                                          int32_t dim);
    __aicore__ inline void WriteBackStateSpec(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start,
                                              int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void DrainTaskMte3();
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();
    __aicore__ inline int32_t FindVarlenSeqByToken(int32_t tokenIdx) const;
    __aicore__ inline bool ResolveExplicitTokenTileSeqRange(int32_t tokenTileId, int32_t &startSeq, int32_t &endSeq) const;
    __aicore__ inline bool ResolveSeqTaskWindow(int32_t seq, int32_t inputMode, int32_t seqLen, int32_t &start,
                                                int32_t &len) const;
    template <int32_t kWindowMode>
    __aicore__ inline bool ResolveSeqTaskWindowByMode(int32_t seq, int32_t seqLen, int32_t &start, int32_t &len) const;
    __aicore__ inline bool ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices, int32_t &cacheIdx) const;
    __aicore__ inline bool ResolveSeqHasInit(int32_t seq, bool hasInitialStateMode) const;
    __aicore__ inline void MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                           int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                           int32_t baseDim, int32_t dim);
    __aicore__ inline void ProcessDefault();
    template <int32_t kWindowMode>
    __aicore__ inline void ProcessDefaultByWindowMode();
    __aicore__ inline void ProcessVarlenTokenTiled();
    __aicore__ inline void CopyInFnChunk(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t chunkStart,
                                         int32_t chunkLen, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void ComputeFnChunk(int32_t chunkStart, int32_t chunkLen, int32_t channelStart, int32_t baseDim,
                                          int32_t dim);
    __aicore__ inline void CopyOutFnChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart, int32_t seqLen,
                                          int32_t cacheIdx, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void ProcessFnChunk(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t seqLen,
                                          int32_t chunkStart, int32_t chunkLen, int32_t channelStart,
                                          int32_t baseDim, int32_t dim);
    __aicore__ inline const CausalConv1dTilingData *GetTilingData() const;
    __aicore__ inline bool HasActivation() const;
    __aicore__ inline bool HasBias() const;
    __aicore__ inline bool IsUpdateSpecDecodingEnabled() const;
    __aicore__ inline bool HasExplicitFnTokenSeqRanges() const;

protected:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;

    TEventID weightBiasMte2ToVEvent_;
    TEventID stateMte2ToVEvent_;
    TEventID inputMte2ToVEvent_[RING_SLOTS];
    TEventID inputVToMte2Event_;
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];
    TEventID stateWritebackMte3ToVEvent_;
    TEventID stateWritebackMte3ToMte2Event_;
    TEventID stateShiftMte2ToMte3Event_;
    TEventID stateShiftVToMte3Event_;
    TEventID stateShiftMte3ToMte2Event_;
    TEventID specWritebackMte2ToMte3Event_[2];
    TEventID specWritebackMte3ToMte2Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int64_t> queryStartLocGm;
    GlobalTensor<int64_t> cacheIndicesGm;
    GlobalTensor<int64_t> initialStateModeGm;
    GlobalTensor<int64_t> numAcceptedTokensGm;
    GlobalTensor<T> yGm;

    const CausalConv1dTilingData *tilingData_{nullptr};
    bool weightCacheValid_{false};
    int32_t cachedC0_{-1};
    int32_t cachedDimTileSize_{-1};
};

template <typename T>
__aicore__ inline void CausalConv1d<T>::ResetRuntimeState(const CausalConv1dTilingData *tilingData)
{
    tilingData_ = tilingData;
    weightCacheValid_ = false;
    cachedC0_ = -1;
    cachedDimTileSize_ = -1;
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::InitSharedBuffersAndEvents()
{
    pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 4) * MAX_BLOCK_DIM * sizeof(float));
    AllocEvents();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::AllocEvents()
{
    weightBiasMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    stateMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    for (int32_t i = 0; i < RING_SLOTS; ++i) {
        inputMte2ToVEvent_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    }
    inputVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    stateWritebackMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    stateWritebackMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    stateShiftMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    stateShiftVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    stateShiftMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    specWritebackMte2ToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    specWritebackMte2ToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    specWritebackMte3ToMte2Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    specWritebackMte3ToMte2Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    for (int32_t i = 0; i < RING_SLOTS; ++i) {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_[i]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(inputVToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::LoadWeightAndBias(int32_t channelStart, int32_t baseDim)
{
    const int32_t dim = tilingData_->dim;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t jStart = MAX_WIDTH - width;
    const bool hasBias = HasBias();
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<T> weightT;
    LocalTensor<T> biasT;

    if constexpr (!std::is_same<T, float>::value) {
        weightT = weightF.ReinterpretCast<T>();
        biasT = biasF.ReinterpretCast<T>();
    }

    for (int32_t j = 0; j < jStart; ++j) {
        Duplicate(weightF[j * MAX_BLOCK_DIM], 0.0f, baseDim);
    }

    for (int32_t j = 0; j < width; ++j) {
        const int32_t jDst = jStart + j;
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + channelStart;
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(weightF[jDst * MAX_BLOCK_DIM], weightGm[weightOffset], baseDim);
        } else {
            DataCopy(weightT[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM], weightGm[weightOffset], baseDim);
        }
    }

    if (hasBias) {
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(biasF, biasGm[channelStart], baseDim);
        } else {
            DataCopy(biasT[MAX_BLOCK_DIM], biasGm[channelStart], baseDim);
        }
    }

    SetFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
    WaitFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);

    if constexpr (!std::is_same<T, float>::value) {
        for (int32_t j = 0; j < width; ++j) {
            const int32_t jDst = jStart + j;
            Cast(weightF[jDst * MAX_BLOCK_DIM], weightT[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM],
                 RoundMode::CAST_NONE, baseDim);
        }
        if (hasBias) {
            Cast(biasF, biasT[MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
        }
        PipeBarrier<PIPE_V>();
    }

    if (!hasBias) {
        Duplicate(biasF, 0.0f, baseDim);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                                 int32_t start, int32_t len, int32_t channelStart,
                                                 int32_t baseDim, int32_t dim)
{
    const int32_t stateLen = tilingData_->stateLen;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t ringStart = MAX_WIDTH - width;
    LocalTensor<T> ring = inBuf.Get<T>();

    for (int32_t i = 0; i < ringStart; ++i) {
        Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
    }
    if (ringStart > 0) {
        PipeBarrier<PIPE_V>();
    }

    if (hasInit) {
        for (int32_t i = 0; i < (width - 1); ++i) {
            const int32_t pos = stateTokenOffset + i;
            const int64_t stateOffset =
                static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(pos) * dim + channelStart;
            DataCopy(ring[(ringStart + i) * MAX_BLOCK_DIM], convStatesGm[stateOffset], baseDim);
        }
        SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
    } else {
        for (int32_t i = 0; i < (width - 1); ++i) {
            Duplicate(ring[(ringStart + i) * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
        }
        PipeBarrier<PIPE_V>();
    }

    if (len > 0) {
        const int32_t slot0 = SlotCurr(0);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + channelStart;
        DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], baseDim);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
    }

    if (len > 1) {
        SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::RunSeq(int32_t start, int32_t len, int32_t channelStart,
                                               int32_t baseDim, int32_t dim)
{
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t jStart = MAX_WIDTH - width;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> accF = biasF[MAX_BLOCK_DIM];
    LocalTensor<float> tmpF = accF[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasBias = HasBias();
    const bool hasActivation = HasActivation();

    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);

        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotCurr]);

        if (t + 1 < len) {
            const int32_t slotNext = SlotPrefetch(t);
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
            WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
            DataCopy(ring[slotNext * MAX_BLOCK_DIM], xGm[xOffsetNext], baseDim);
            SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotNext]);
        }

        bool accInitialized = false;
        if (hasBias) {
            Adds(accF, biasF, 0.0f, baseDim);
            PipeBarrier<PIPE_V>();
            accInitialized = true;
        }

        for (int32_t j = jStart; j < MAX_WIDTH; ++j) {
            const int32_t tap = (MAX_WIDTH - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            if (!accInitialized) {
                Mul(accF, tmpF, weightF[j * MAX_BLOCK_DIM], baseDim);
                accInitialized = true;
            } else {
                MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], baseDim);
            }
        }

        if (hasActivation) {
            Silu(tmpF, accF, baseDim);
        } else {
            PipeBarrier<PIPE_V>();
        }

        const int32_t outSlot = t & 1;
        LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
        if (t >= 2) {
            WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outSlotT, tmpF, baseDim);
            } else {
                DataCopy(outSlotT, accF, baseDim);
            }
        } else {
            if (hasActivation) {
                Cast(outSlotT, tmpF, RoundMode::CAST_RINT, baseDim);
            } else {
                Cast(outSlotT, accF, RoundMode::CAST_RINT, baseDim);
            }
        }

        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        DataCopy(yGm[outOffset], outSlotT, baseDim);
        if (t + 2 < len) {
            SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if (t + 2 < len) {
            SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
        }
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::DrainTaskMte3()
{
    SetFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    WaitFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
    SetFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
    WaitFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart,
                                                       int32_t baseDim, int32_t dim)
{
    const int32_t stateLen = tilingData_->stateLen;
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    if (len <= 0) {
        return;
    }

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();
    const int32_t lastSlot = SlotCurr(lastT);
    const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim + channelStart;

    for (int32_t pos = 0; pos < (width - 1); ++pos) {
        const int32_t tap = (width - 2) - pos;
        const int32_t slot = RetreatRingSlot(lastSlot, tap);
        const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(pos) * dim;
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], baseDim);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackStateSpec(int32_t cacheIdx, bool hasInit,
                                                           int32_t stateTokenOffset, int32_t start, int32_t len,
                                                           int32_t channelStart, int32_t baseDim, int32_t dim)
{
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const int32_t stateLen = tilingData_->stateLen;
    if (len <= 0) {
        return;
    }

    if (width != 4) {
        WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
        return;
    }

    constexpr int32_t keep = MAX_WIDTH - 2;
    const int32_t reqStateLen = keep + len;
    if (reqStateLen > stateLen) {
        WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
        return;
    }

    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> buf0 = ring[0 * MAX_BLOCK_DIM];
    LocalTensor<T> buf1 = ring[1 * MAX_BLOCK_DIM];

    if (hasInit) {
        const int32_t srcPos0 = stateTokenOffset + 1;
        const int32_t srcPos1 = stateTokenOffset + 2;
        const int64_t srcOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos0) * dim + channelStart;
        const int64_t srcOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos1) * dim + channelStart;
        DataCopy(buf0, convStatesGm[srcOffset0], baseDim);
        DataCopy(buf1, convStatesGm[srcOffset1], baseDim);
        SetFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        const int64_t dstOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + channelStart;
        const int64_t dstOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + channelStart;
        DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
        DataCopy(convStatesGm[dstOffset1], buf1, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    } else {
        Duplicate(buf0, static_cast<T>(0), baseDim);
        SetFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        const int64_t dstOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + channelStart;
        const int64_t dstOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + channelStart;
        DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
        DataCopy(convStatesGm[dstOffset1], buf0, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
    }

    const int64_t xOffset0 = static_cast<int64_t>(start) * dim + channelStart;
    DataCopy(buf0, xGm[xOffset0], baseDim);
    SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);

    for (int32_t t = 0; t < len; ++t) {
        const int32_t curr = t & 1;
        const int32_t next = curr ^ 1;
        LocalTensor<T> currBuf = (curr == 0) ? buf0 : buf1;
        LocalTensor<T> nextBuf = (next == 0) ? buf0 : buf1;

        WaitFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[curr]);

        if (t + 1 < len) {
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
            if (t > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[next]);
            }
            DataCopy(nextBuf, xGm[xOffsetNext], baseDim);
            SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[next]);
        }

        const int64_t dstOffset =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(keep + t) * dim + channelStart;
        DataCopy(convStatesGm[dstOffset], currBuf, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[curr]);
    }

    WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
    if (len > 1) {
        WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
    }
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::ResolveSeqTaskWindow(int32_t seq, int32_t inputMode, int32_t seqLen,
                                                             int32_t &start, int32_t &len) const
{
    switch (GetSeqTaskWindowMode(inputMode)) {
        case SEQ_TASK_WINDOW_MODE_VARLEN:
            return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_VARLEN>(seq, seqLen, start, len);
        case SEQ_TASK_WINDOW_MODE_DECODE2D:
            return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_DECODE2D>(seq, seqLen, start, len);
        default:
            return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_BATCH>(seq, seqLen, start, len);
    }
}

template <typename T>
template <int32_t kWindowMode>
__aicore__ inline bool CausalConv1d<T>::ResolveSeqTaskWindowByMode(int32_t seq, int32_t seqLen, int32_t &start,
                                                                   int32_t &len) const
{
    SeqTaskWindow window;
    if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_VARLEN) {
        const int32_t startVal = queryStartLocGm.GetValue(seq);
        const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
        window = BuildSeqTaskWindowVarlen(startVal, endVal);
    } else if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_DECODE2D) {
        window = BuildSeqTaskWindowDecode2D(seq);
    } else {
        window = BuildSeqTaskWindowBatch(seq, seqLen);
    }

    if (!window.valid) {
        return false;
    }
    start = window.start;
    len = window.len;
    return true;
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices,
                                                             int32_t &cacheIdx) const
{
    cacheIdx = seq;
    if (!hasCacheIndices) {
        return true;
    }

    const int64_t cacheIdx64 = cacheIndicesGm.GetValue(seq);
    if (cacheIdx64 == tilingData_->padSlotId) {
        return false;
    }
    cacheIdx = static_cast<int32_t>(cacheIdx64);
    return true;
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::ResolveSeqHasInit(int32_t seq, bool hasInitialStateMode) const
{
    if (tilingData_->runMode == CAUSAL_CONV1D_RUN_MODE_UPDATE) {
        return true;
    }
    return hasInitialStateMode ? (initialStateModeGm.GetValue(seq) != 0) : false;
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ProcessDefault()
{
    switch (GetSeqTaskWindowMode(tilingData_->inputMode)) {
        case SEQ_TASK_WINDOW_MODE_VARLEN:
            ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_VARLEN>();
            return;
        case SEQ_TASK_WINDOW_MODE_DECODE2D:
            ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_DECODE2D>();
            return;
        default:
            ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_BATCH>();
            return;
    }
}

template <typename T>
template <int32_t kWindowMode>
__aicore__ inline void CausalConv1d<T>::ProcessDefaultByWindowMode()
{
    const int32_t dim = tilingData_->dim;
    const int32_t batch = tilingData_->batch;
    const int32_t seqLen = tilingData_->seqLen;
    const int32_t baseDim = static_cast<int32_t>(tilingData_->baseDim);
    const int32_t baseDimCnt = static_cast<int32_t>(tilingData_->baseDimCnt);
    const int32_t width = static_cast<int32_t>(tilingData_->width);
    const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
    const bool hasInitialStateMode = (tilingData_->hasInitialStateMode != 0);
    const bool isSpecDecodingGlobal = IsUpdateSpecDecodingEnabled();

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (baseDim <= 0 || baseDimCnt <= 0 || baseDim > MAX_BLOCK_DIM || width < 2 || width > MAX_WIDTH) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * baseDimCnt;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / baseDimCnt);
        const int32_t baseDimIdx = static_cast<int32_t>(task % baseDimCnt);
        const int32_t channelStart = baseDimIdx * baseDim;
        if (channelStart >= dim) {
            continue;
        }
        const int32_t curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);

        int32_t start = 0;
        int32_t len = 0;
        if (!ResolveSeqTaskWindowByMode<kWindowMode>(seq, seqLen, start, len)) {
            continue;
        }

        int32_t cacheIdx = 0;
        if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
            continue;
        }

        const bool hasInit = ResolveSeqHasInit(seq, hasInitialStateMode);
        int32_t stateTokenOffset = 0;
        if (isSpecDecodingGlobal) {
            int32_t accepted = static_cast<int32_t>(numAcceptedTokensGm.GetValue(seq));
            stateTokenOffset = accepted - 1;
            const int32_t maxOffset = static_cast<int32_t>(tilingData_->stateLen - (width - 1));
            if (stateTokenOffset < 0) {
                stateTokenOffset = 0;
            } else if (stateTokenOffset > maxOffset) {
                stateTokenOffset = maxOffset;
            }
        }

        const bool weightCacheHit =
            weightCacheValid_ && (cachedC0_ == channelStart) && (cachedDimTileSize_ == curBaseDim);
        if (!weightCacheHit) {
            LoadWeightAndBias(channelStart, curBaseDim);
            weightCacheValid_ = true;
            cachedC0_ = channelStart;
            cachedDimTileSize_ = curBaseDim;
        }

        InitRing(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
        RunSeq(start, len, channelStart, curBaseDim, dim);

        if (isSpecDecodingGlobal) {
            DrainTaskMte3();
            WriteBackStateSpec(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
        } else {
            WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim);
        }

        DrainTaskMte3();
    }
}

template <typename T>
__aicore__ inline const CausalConv1dTilingData *CausalConv1d<T>::GetTilingData() const
{
    return tilingData_;
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::HasActivation() const
{
    return (tilingData_ != nullptr) && (tilingData_->activationMode != 0);
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::HasBias() const
{
    return (tilingData_ != nullptr) && (tilingData_->hasBias != 0);
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::IsUpdateSpecDecodingEnabled() const
{
    return (tilingData_->runMode == CAUSAL_CONV1D_RUN_MODE_UPDATE) &&
           (tilingData_->hasNumAcceptedTokens != 0) && (tilingData_->width == 4);
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::HasExplicitFnTokenSeqRanges() const
{
    return (tilingData_ != nullptr) && (tilingData_->inputMode == 0) &&
           (tilingData_->hasExplicitTokenSeqRanges != 0) &&
           (tilingData_->explicitTokenSeqRangeCount >= tilingData_->tokenBlockCnt);
}

#include "causal_conv1d_fn_tasks.h"

}  // namespace NsCausalConv1d
#endif  // CAUSAL_CONV1D_H
