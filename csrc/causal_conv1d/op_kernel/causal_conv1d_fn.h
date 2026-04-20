#ifndef CAUSAL_CONV1D_FN_H
#define CAUSAL_CONV1D_FN_H

#include "causal_conv1d.h"

namespace NsCausalConv1d {

template <typename T>
class CausalConv1dFn : public CausalConv1d<T> {
public:
    __aicore__ inline CausalConv1dFn() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                GM_ADDR convStateIndices, GM_ADDR queryStartLoc, GM_ADDR numAcceptedTokens,
                                GM_ADDR initialState, GM_ADDR y, GM_ADDR workspace,
                                const CausalConv1dTilingData *tilingData)
    {
        this->ResetRuntimeState(tilingData);

        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
        this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
        this->cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(convStateIndices));
        this->queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
        this->numAcceptedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(numAcceptedTokens));
        this->initialStateModeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initialState));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

        (void)workspace;

        this->InitSharedBuffersAndEvents();
    }

    __aicore__ inline void Process()
    {
        if (this->GetTilingData()->baseDimCnt > 1) {
            this->ProcessDefault();
        } else {
            this->ProcessVarlenTokenTiled();
        }
        this->ReleaseEvents();
    }
};

template <typename T>
__aicore__ inline void RunCausalConv1dFn(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                         GM_ADDR convStateIndices, GM_ADDR queryStartLoc,
                                         GM_ADDR numAcceptedTokens, GM_ADDR initialState,
                                         GM_ADDR y, GM_ADDR workspace,
                                         const CausalConv1dTilingData *tilingData)
{
    CausalConv1dFn<T> op;
    op.Init(x, weight, bias, convStates, convStateIndices, queryStartLoc, numAcceptedTokens, initialState, y, workspace,
            tilingData);
    op.Process();
}

}  // namespace NsCausalConv1d

#endif  // CAUSAL_CONV1D_FN_H
