/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file np_fft-inl.h
 * \brief Function definition of numpy-compatible fft operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_FFT_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_FFT_INL_H_

#include <vector>
#include <string>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../tensor/broadcast_reduce_op.h"

#if MXNET_USE_CUDA
#include <cufft.h>
#endif

namespace mxnet {
namespace op {

struct NPFFTParam : public dmlc::Parameter<NPFFTParam> {
  int axis, compute_size;  // the maximum size of sub-batch to be forwarded through FFT in one time
  dmlc::optional<int> n;
  dmlc::optional<std::string> norm;
  DMLC_DECLARE_PARAMETER(NPFFTParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
    .describe("Axis over which to compute the FFT");
    DMLC_DECLARE_FIELD(compute_size).set_default(128)
    .describe("Maximum size of sub-batch to be forwarded at one time");
  }
};

template <typename xpu, typename IType, typename OType>
inline void FFTExec(const OpContext& ctx,
                      const mshadow::Tensor<xpu, 2, IType>& in_data,
                      const mshadow::Tensor<xpu, 2, OType>& out_data,
                      int n_ffts, int dim_, int stride_, size_t num_compute, int compute_size) {
  using namespace mshadow;

  Stream<xpu>* s = ctx.get_stream<xpu>();
  // Tensor<xpu, 1, IType> workspace =
  //           ctx.requested[0].get_space_typed<xpu, 1, IType>(
  //               Shape1(compute_size*dim_), s);
  // Tensor<xpu, 2, IType> input_buffer = Tensor<xpu, 2, IType>(workspace.dptr_,
  //                           Shape2(compute_size, dim_), s);
  Tensor<xpu, 2, OType> input_buffer =
    ctx.requested[0].get_space_typed<xpu, 2, OType>(Shape2(compute_size, dim_), s);
  // start fft
  cufftHandle plan;
  cufftPlanMany(&plan, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, compute_size);
  for (size_t idx=0; idx < num_compute; ++idx) {
    // input_buffer = complex_pad_imag(in_data.Slice(idx*compute_size, idx*compute_size+compute_size));
    // input_buffer = in_data.Slice(idx*compute_size, idx*compute_size+compute_size);
    cufftComplex* in_tmp = const_cast<cufftComplex*>(
      reinterpret_cast<const cufftComplex*>(input_buffer.dptr_));
    cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out_data.dptr_ + idx*stride_);
    CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
  }
  cufftDestroy(plan);

  // handle the remaining samples
  size_t remain_num = n_ffts - compute_size*num_compute;
  if (remain_num > 0) {
    cufftHandle plan_remain;
    cufftPlanMany(&plan_remain, 1, &dim_, nullptr, 0, 0, nullptr, 0, 0,
                  CUFFT_C2C, remain_num);

    // for (size_t entry = 0; entry < remain_num; entry++){
    //   input_buffer[entry] = OType(in_data.dptr_[entry]);
    //   // input_buffer[entry] = OType(in_data.Slice(num_compute*compute_size, num_compute*compute_size+remain_num)[entry]);
    // }
    long int x = 1;
    // int x = in_data.dptr_[0];
    input_buffer[0] = OType(x);
    // input_buffer = in_data.Slice(num_compute*compute_size, num_compute*compute_size+remain_num);
    cufftComplex* in_tmp = const_cast<cufftComplex*>(
      reinterpret_cast<const cufftComplex*>(input_buffer.dptr_));
    cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out_data.dptr_ + num_compute*stride_);
    CHECK_EQ(cufftExecC2C(plan_remain, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
    cufftDestroy(plan_remain);
  }
}

template <typename xpu>
void FFTForwardImpl(const OpContext& ctx, const TBlob& in, const TBlob& out,
                     const int axis, const int compute_size) {
  using namespace mshadow;
  using namespace mxnet_op;

  int axis_checked = CheckAxis(axis, in.ndim());
  // stride for elements on the given axis, same in input and output
  int stride = 1;
  for (int i = in.ndim() - 1; i > axis_checked; --i) {
    stride *= in.shape_[i];
  }

  int n_ffts = in.shape_.ProdShape(0, in.ndim()-1);
  int dim_ = in.shape_[in.ndim()-1];
  int stride_ = compute_size*dim_;
  int num_compute = n_ffts / compute_size;

  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(in.type_flag_, IType, {
    MSHADOW_COMPLEX_TYPE_SWITCH(out.type_flag_, OType, {
      Tensor<xpu, 2, IType> in_data = in.get_with_shape<xpu, 2, IType>(
        Shape2(n_ffts, dim_), s);
      Tensor<xpu, 2, OType> out_data = out.get_with_shape<xpu, 2, OType>(
        Shape2(n_ffts, dim_), s);
      FFTExec<xpu, IType, OType>(ctx, in_data, out_data, n_ffts, dim_, stride_, num_compute, compute_size);
    });
  });
}

template <typename xpu>
void FFTForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const NPFFTParam& param = nnvm::get<NPFFTParam>(attrs.parsed);

  FFTForwardImpl<xpu>(ctx, inputs[0], outputs[0], param.axis, param.compute_size);
}


template <typename xpu>
void FFTBackwardImpl(const OpContext& ctx, const TBlob& in, const TBlob& out,
                     const int axis, const int compute_size) {
  using namespace mshadow;
  using namespace mxnet_op;

  int axis_checked = CheckAxis(axis, in.ndim());
  // stride for elements on the given axis, same in input and output
  int stride = 1;
  for (int i = in.ndim() - 1; i > axis_checked; --i) {
    stride *= in.shape_[i];
  }

  int n_ffts = in.shape_.ProdShape(0, in.ndim()-1);
  int dim_ = in.shape_[in.ndim()-1];
  int stride_ = compute_size*dim_;
  int num_compute = n_ffts / compute_size;

  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(in.type_flag_, IType, {
    MSHADOW_COMPLEX_TYPE_SWITCH(out.type_flag_, OType, {
      Tensor<xpu, 2, IType> in_data = in.get_with_shape<xpu, 2, IType>(
        Shape2(n_ffts, dim_), s);
      Tensor<xpu, 2, OType> out_data = out.get_with_shape<xpu, 2, OType>(
        Shape2(n_ffts, dim_), s);
      // Tensor<xpu, 2, OType> input_buffer =
      //   ctx.requested[0].get_space_typed<xpu, 2, OType>(Shape2(compute_size, dim_), s);
      FFTExec<xpu, IType, OType>(ctx, in_data, out_data, n_ffts, dim_, stride_, num_compute, compute_size);
    });
  });
}

template <typename xpu>
void FFTBackward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const NPFFTParam& param = nnvm::get<NPFFTParam>(attrs.parsed);

  FFTBackwardImpl<xpu>(ctx, inputs[0], outputs[0], param.axis, param.compute_size);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_FFT_INL_H_
