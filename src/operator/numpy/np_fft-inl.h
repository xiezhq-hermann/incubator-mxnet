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
  int batch_size;  // the maximum size of sub-batch to be forwarded through FFT in one time
  dmlc::optional<int> n;
  dmlc::optional<int> axis;
  dmlc::optional<std::string> norm;
  DMLC_DECLARE_PARAMETER(NPFFTParam) {
    DMLC_DECLARE_FIELD(n).set_default(dmlc::optional<int>())
    .describe("Length of the transformed axis of the output");
    DMLC_DECLARE_FIELD(batch_size).set_default(128)
    .describe("Maximum size of sub-batch to be forwarded at one time");
  }
};

struct resize_and_cast {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, OType* out, IType* in, int dim, int n) {
    int tmp = i / n;
    int offset = i - tmp;
    out[i] = (offset >= dim) ? OType(0) : OType(in[tmp*dim+offset]);
  }
};

template <typename xpu, typename IType, typename OType>
inline void FFTExec(const OpContext& ctx,
                      const mshadow::Tensor<xpu, 2, IType>& in_data,
                      const mshadow::Tensor<xpu, 2, OType>& out_data,
                      int n_ffts, int len_fft, int batch_size) {
  using namespace mshadow;
  using namespace mxnet_op;
  
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, OType> input_buffer =
    ctx.requested[0].get_space_typed<xpu, 2, OType>(Shape2(batch_size, len_fft), s);
  
  // start fft
  cufftHandle plan;
  cufftPlanMany(&plan, 1, &len_fft, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, batch_size);
  size_t num_compute = n_ffts / batch_size;
  int stride_ = batch_size * len_fft;

  for (size_t idx=0; idx < num_compute; ++idx) {
    Kernel<resize_and_cast, xpu>::Launch(
        s, stride_, input_buffer.dptr_,
        in_data.Slice(idx*batch_size, idx*batch_size+batch_size).dptr_,
        in_data.shape_[1], len_fft);
    cufftComplex* in_tmp = const_cast<cufftComplex*>(
      reinterpret_cast<const cufftComplex*>(input_buffer.dptr_));
    cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out_data.dptr_ + idx*stride_);
    CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
  }
  cufftDestroy(plan);

  // handle the remaining samples
  size_t remain_num = n_ffts - batch_size*num_compute;
  if (remain_num > 0) {
    cufftHandle plan_remain;
    cufftPlanMany(&plan_remain, 1, &len_fft, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, remain_num);

    Kernel<resize_and_cast, xpu>::Launch(
        s, remain_num*len_fft, input_buffer.dptr_,
        in_data.Slice(num_compute*batch_size, num_compute*batch_size+remain_num).dptr_,
        in_data.shape_[1], len_fft);
    cufftComplex* in_tmp = const_cast<cufftComplex*>(
      reinterpret_cast<const cufftComplex*>(input_buffer.dptr_));
    cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out_data.dptr_ + num_compute*stride_);
    CHECK_EQ(cufftExecC2C(plan_remain, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
    cufftDestroy(plan_remain);
  }
}

template <typename xpu>
void FFTForwardImpl(const OpContext& ctx, const TBlob& in, const TBlob& out,
                     const dmlc::optional<int>& n, const int batch_size) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK(!n.has_value() || n.value() > 0);
  if (out.Size() == 0) return;

  int n_ffts = in.shape_.ProdShape(0, in.ndim()-1);
  int len_fft = out.shape_[in.ndim()-1];

  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_COMPLEX(in.type_flag_, IType, {
    MSHADOW_COMPLEX_TYPE_SWITCH(out.type_flag_, OType, {
      Tensor<xpu, 2, IType> in_data = in.get_with_shape<xpu, 2, IType>(
        Shape2(n_ffts, in.shape_[in.ndim()-1]), s);
      Tensor<xpu, 2, OType> out_data = out.get_with_shape<xpu, 2, OType>(
        Shape2(n_ffts, len_fft), s);
      FFTExec<xpu, IType, OType>(ctx, in_data, out_data, n_ffts, len_fft, batch_size);
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

  FFTForwardImpl<xpu>(ctx, inputs[0], outputs[0], param.n, param.batch_size);
}


template <typename xpu>
void FFTBackwardImpl(const OpContext& ctx, const TBlob& ograd, const TBlob& igrad,
                     const dmlc::optional<int>& n, const int batch_size) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK(!n.has_value() || n.value() > 0);
  if (igrad.Size() == 0) return;

  int n_ffts = ograd.shape_.ProdShape(0, ograd.ndim()-1);
  int len_fft = ograd.shape_[ograd.ndim()-1];

  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_COMPLEX(ograd.type_flag_, IType, {
    MSHADOW_COMPLEX_TYPE_SWITCH(igrad.type_flag_, OType, {
      Tensor<xpu, 2, IType> in_data = ograd.get_with_shape<xpu, 2, IType>(
        Shape2(n_ffts, len_fft), s);
      Tensor<xpu, 2, OType> out_data = igrad.get_with_shape<xpu, 2, OType>(
        Shape2(n_ffts, len_fft), s);
      FFTExec<xpu, IType, OType>(ctx, in_data, out_data, n_ffts, len_fft, batch_size);
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

  FFTBackwardImpl<xpu>(ctx, inputs[0], outputs[0], param.n, param.batch_size);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_FFT_INL_H_
