/*!
 * \file causal_conv1d.h
 * \brief causal_conv1d host-side function declaration
 */

#ifndef CAUSAL_CONV1D_HOST_H_
#define CAUSAL_CONV1D_HOST_H_

#include <ATen/ATen.h>
#include "defines.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor causal_conv1d_impl(const at::Tensor &x, const at::Tensor &weight,
                                       const at::Tensor &bias, const at::Tensor &conv_state,
                                       const at::Tensor &conv_state_indices, const at::Tensor &query_start_loc,
                                       const at::Tensor &num_accepted_tokens, const at::Tensor &initial_state,
                                       bool activation_mode, int64_t pad_slot_id, int64_t run_mode);

}  // namespace npu_kernel
}  // namespace sglang

#endif  // CAUSAL_CONV1D_HOST_H_
