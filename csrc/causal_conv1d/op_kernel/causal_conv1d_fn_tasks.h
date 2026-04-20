#ifndef CAUSAL_CONV1D_FN_TASKS_H
#define CAUSAL_CONV1D_FN_TASKS_H

template <typename T>
__aicore__ inline int32_t CausalConv1d<T>::FindVarlenSeqByToken(int32_t tokenIdx) const
{
    int32_t left = 0;
    int32_t right = static_cast<int32_t>(tilingData_->batch);
    while (left < right) {
        const int32_t mid = left + ((right - left) >> 1);
        const int32_t endVal = static_cast<int32_t>(queryStartLocGm.GetValue(mid + 1));
        if (tokenIdx < endVal) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

template <typename T>
__aicore__ inline bool CausalConv1d<T>::ResolveExplicitTokenTileSeqRange(int32_t tokenTileId, int32_t &startSeq,
                                                                         int32_t &endSeq) const
{
    if (!HasExplicitFnTokenSeqRanges()) {
        return false;
    }
    if (tokenTileId < 0 || tokenTileId >= tilingData_->explicitTokenSeqRangeCount) {
        return false;
    }
    startSeq = static_cast<int32_t>(tilingData_->tokenTileStartSeq[tokenTileId]);
    endSeq = static_cast<int32_t>(tilingData_->tokenTileEndSeq[tokenTileId]);
    return true;
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::CopyInFnChunk(int32_t cacheIdx, bool hasInit, int32_t seqStart,
                                                      int32_t chunkStart, int32_t chunkLen, int32_t channelStart,
                                                      int32_t baseDim, int32_t dim)
{
    InitRing(cacheIdx, hasInit, 0, chunkStart, chunkLen, channelStart, baseDim, dim);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ComputeFnChunk(int32_t chunkStart, int32_t chunkLen, int32_t channelStart,
                                                       int32_t baseDim, int32_t dim)
{
    RunSeq(chunkStart, chunkLen, channelStart, baseDim, dim);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::CopyOutFnChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                       int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                       int32_t baseDim, int32_t dim)
{
    if (chunkStart + chunkLen == seqStart + seqLen) {
        WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim);
    }
    DrainTaskMte3();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ProcessFnChunk(int32_t cacheIdx, bool hasInit, int32_t seqStart,
                                                       int32_t seqLen, int32_t chunkStart, int32_t chunkLen,
                                                       int32_t channelStart, int32_t baseDim, int32_t dim)
{
    CopyInFnChunk(cacheIdx, hasInit, seqStart, chunkStart, chunkLen, channelStart, baseDim, dim);
    ComputeFnChunk(chunkStart, chunkLen, channelStart, baseDim, dim);
    CopyOutFnChunk(chunkStart, chunkLen, seqStart, seqLen, cacheIdx, channelStart, baseDim, dim);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen,
                                                                        int32_t seqStart, int32_t seqLen,
                                                                        int32_t cacheIdx, int32_t channelStart,
                                                                        int32_t baseDim, int32_t dim)
{
    if (chunkStart + chunkLen == seqStart + seqLen) {
        WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim);
    }
    DrainTaskMte3();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ProcessVarlenTokenTiled()
{
    const int32_t dim = tilingData_->dim;
    const int32_t batch = tilingData_->batch;
    const int32_t seqLen = tilingData_->seqLen;
    const int32_t cuSeqlen = tilingData_->cuSeqlen;
    const int32_t baseDim = static_cast<int32_t>(tilingData_->baseDim);
    const int32_t baseDimCnt = static_cast<int32_t>(tilingData_->baseDimCnt);
    const int32_t tokenBlockSize = static_cast<int32_t>(tilingData_->tokenBlockSize);
    const int32_t tokenBlockCnt = static_cast<int32_t>(tilingData_->tokenBlockCnt);
    const int32_t inputMode = static_cast<int32_t>(tilingData_->inputMode);
    const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
    const bool hasInitialStateMode = (tilingData_->hasInitialStateMode != 0);

    const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());

    if (baseDim <= 0 || baseDimCnt <= 0 || baseDim > MAX_BLOCK_DIM || tokenBlockSize <= 0 || tokenBlockCnt <= 0 ||
        dim <= 0 || cuSeqlen <= 0) {
        ReleaseEvents();
        return;
    }

    const int64_t phase1Grid = static_cast<int64_t>(tokenBlockCnt) * baseDimCnt;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < phase1Grid; task += static_cast<int64_t>(blockNum)) {
        const int32_t tokenTileId = static_cast<int32_t>(task / baseDimCnt);
        const int32_t dimBlockId = static_cast<int32_t>(task % baseDimCnt);
        const int32_t channelStart = dimBlockId * baseDim;
        if (channelStart >= dim) {
            continue;
        }
        const int32_t curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);

        const int32_t tokenStart = tokenTileId * tokenBlockSize;
        if (tokenStart >= cuSeqlen) {
            continue;
        }
        const int32_t tokenEndRaw = tokenStart + tokenBlockSize;
        const int32_t tokenEnd = (tokenEndRaw <= cuSeqlen) ? tokenEndRaw : cuSeqlen;
        if (tokenEnd <= tokenStart) {
            continue;
        }

        int32_t seq = 0;
        int32_t seqUpperBound = batch;
        if (inputMode == 0) {
            int32_t startSeq = 0;
            int32_t endSeq = 0;
            if (ResolveExplicitTokenTileSeqRange(tokenTileId, startSeq, endSeq)) {
                seq = startSeq;
                seqUpperBound = endSeq;
            } else {
                seq = FindVarlenSeqByToken(tokenStart);
            }
        } else {
            seq = (seqLen > 0) ? (tokenStart / seqLen) : 0;
        }

        int32_t cursor = tokenStart;
        while (cursor < tokenEnd && seq < seqUpperBound) {
            int32_t seqStart = 0;
            int32_t curSeqLen = 0;

            if (inputMode == 0) {
                const int32_t cuSeqlenVal = tilingData_->cuSeqlen;
                seqStart = static_cast<int32_t>(queryStartLocGm.GetValue(seq));
                curSeqLen = (seq + 1 < batch)
                                ? (static_cast<int32_t>(queryStartLocGm.GetValue(seq + 1)) - seqStart)
                                : (cuSeqlenVal - seqStart);
            } else {
                curSeqLen = seqLen;
                seqStart = seq * seqLen;
            }

            if (curSeqLen <= 0) {
                ++seq;
                continue;
            }

            const int32_t curSeqEnd = seqStart + curSeqLen;
            if (cursor < seqStart) {
                cursor = seqStart;
            }
            if (cursor >= curSeqEnd) {
                ++seq;
                continue;
            }

            const int32_t tileEnd = (tokenEnd <= curSeqEnd) ? tokenEnd : curSeqEnd;
            const int32_t tileLen = tileEnd - cursor;
            if (tileLen <= 0) {
                ++seq;
                continue;
            }

            int32_t cacheIdx = 0;
            if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
                cursor = tileEnd;
                ++seq;
                continue;
            }

            const bool hasInit = ResolveSeqHasInit(seq, hasInitialStateMode);

            const bool weightCacheHit =
                weightCacheValid_ && (cachedC0_ == channelStart) && (cachedDimTileSize_ == curBaseDim);
            if (!weightCacheHit) {
                LoadWeightAndBias(channelStart, curBaseDim);
                weightCacheValid_ = true;
                cachedC0_ = channelStart;
                cachedDimTileSize_ = curBaseDim;
            }

            ProcessFnChunk(cacheIdx, hasInit, seqStart, curSeqLen, cursor, tileLen, channelStart, curBaseDim, dim);

            cursor = tileEnd;
            ++seq;
        }
    }

    ReleaseEvents();
}

#endif  // CAUSAL_CONV1D_FN_TASKS_H
