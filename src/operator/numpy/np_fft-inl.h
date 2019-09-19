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

#include <string>
#include <vector>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../tensor/broadcast_reduce_op.h"

#if MXNET_USE_CUDA
#include <cufft.h>
#endif

namespace mxnet {
namespace op {

struct NPFFTParam : public dmlc::Parameter<NPFFTParam> {
  // the maximum size of sub-batch to be forwarded through FFT in one time
  int batch_size;
  dmlc::optional<int> n;
  dmlc::optional<int> axis;
  dmlc::optional<std::string> norm;
  DMLC_DECLARE_PARAMETER(NPFFTParam) {
    DMLC_DECLARE_FIELD(n)
        .set_default(dmlc::optional<int>())
        .describe("Length of the transformed axis of the output");
    DMLC_DECLARE_FIELD(batch_size)
        .set_default(128)
        .describe("Maximum size of sub-batch to be forwarded at one time");
  }
};

struct resize_and_cast {
  template <typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, OType* out, IType* in,
                                  int in_dim, int out_dim, float scale = 1) {
    int tmp = i / out_dim;
    int offset = i - tmp * out_dim;
    out[i] = (offset >= in_dim) ? OType(0)
                                : OType(in[tmp * in_dim + offset] / scale);
  }
};

template <typename xpu>
inline void cuFFTPlan(cufftHandle* plan, int len_fft, int batch_size,
                      const mshadow::complex::complex64& indicator) {
  cufftPlanMany(plan, 1, &len_fft, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, batch_size);
}

template <typename xpu>
inline void cuFFTPlan(cufftHandle* plan, int len_fft, int batch_size,
                      const mshadow::complex::complex128& indicator) {
  cufftPlanMany(plan, 1, &len_fft, nullptr, 0, 0, nullptr, 0, 0, CUFFT_Z2Z, batch_size);
}

template <typename xpu, typename IType, typename OType>
inline void cuFFTExec(
    const OpContext& ctx, const cufftHandle& plan,
    const mshadow::Tensor<xpu, 2, IType>& in_data,
    const mshadow::Tensor<xpu, 2, OType>& out_data,
    const mshadow::Tensor<xpu, 2, mshadow::complex::complex64>& input_buffer,
    size_t offset, int batch_size, int len_fft, int grad_dim) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();

  Kernel<resize_and_cast, xpu>::Launch(
      s, batch_size * len_fft, input_buffer.dptr_,
      in_data.Slice(offset, offset + batch_size).dptr_,
      in_data.shape_[1], len_fft);
  cufftComplex* in_tmp = const_cast<cufftComplex*>(
      reinterpret_cast<const cufftComplex*>(input_buffer.dptr_));
  if (grad_dim == 0) {
    cufftComplex* out_tmp = reinterpret_cast<cufftComplex*>(out_data.dptr_ + offset * len_fft);
    CHECK_EQ(cufftExecC2C(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
  } else {
    CHECK_EQ(cufftExecC2C(plan, in_tmp, in_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);
    Kernel<resize_and_cast, xpu>::Launch(
        s, batch_size * grad_dim,
        out_data.Slice(offset, offset + batch_size).dptr_, input_buffer.dptr_,
        len_fft, grad_dim, len_fft);
  }
}

template <typename xpu, typename IType, typename OType>
inline void cuFFTExec(
    const OpContext& ctx, const cufftHandle& plan,
    const mshadow::Tensor<xpu, 2, IType>& in_data,
    const mshadow::Tensor<xpu, 2, OType>& out_data,
    const mshadow::Tensor<xpu, 2, mshadow::complex::complex128>& input_buffer,
    size_t offset, int batch_size, int len_fft, int grad_dim) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();

  Kernel<resize_and_cast, xpu>::Launch(
      s, batch_size * len_fft, input_buffer.dptr_,
      in_data.Slice(offset, offset + batch_size).dptr_,
      in_data.shape_[1], len_fft);
  cufftDoubleComplex* in_tmp = const_cast<cufftDoubleComplex*>(
      reinterpret_cast<const cufftDoubleComplex*>(input_buffer.dptr_));
  if (grad_dim == 0) {
    cufftDoubleComplex* out_tmp = reinterpret_cast<cufftDoubleComplex*>(out_data.dptr_ + offset * len_fft);
    CHECK_EQ(cufftExecZ2Z(plan, in_tmp, out_tmp, CUFFT_FORWARD), CUFFT_SUCCESS);
  } else {
    CHECK_EQ(cufftExecZ2Z(plan, in_tmp, in_tmp, CUFFT_INVERSE), CUFFT_SUCCESS);
    Kernel<resize_and_cast, xpu>::Launch(
        s, batch_size * grad_dim,
        out_data.Slice(offset, offset + batch_size).dptr_, input_buffer.dptr_,
        len_fft, grad_dim, len_fft);
  }
}

template <typename xpu, typename IType, typename OType, typename FFTType>
inline void FFTExec(const OpContext& ctx,
                    const mshadow::Tensor<xpu, 2, IType>& in_data,
                    const mshadow::Tensor<xpu, 2, OType>& out_data,
                    const FFTType& indicator, int n_ffts, int len_fft,
                    int batch_size, int grad_dim = 0) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, FFTType> input_buffer =
      ctx.requested[0].get_space_typed<xpu, 2, FFTType>(Shape2(batch_size, len_fft), s);

  // start fft
  cufftHandle plan;
  cuFFTPlan<xpu>(&plan, len_fft, batch_size, indicator);

  size_t num_compute = n_ffts / batch_size;
  for (size_t idx = 0; idx < num_compute; ++idx) {
    cuFFTExec(ctx, plan, in_data, out_data, input_buffer, idx * batch_size,
              batch_size, len_fft, grad_dim);
  }
  cufftDestroy(plan);

  // handle the remaining samples
  int remain_num = n_ffts - batch_size * num_compute;
  if (remain_num > 0) {
    cufftHandle plan_remain;
    cuFFTPlan<xpu>(&plan_remain, len_fft, remain_num, indicator);
    cuFFTExec(ctx, plan_remain, in_data, out_data, input_buffer,
              num_compute * batch_size, remain_num, len_fft, grad_dim);
    cufftDestroy(plan_remain);
  }
}

template <typename xpu>
void FFTForwardImpl(const OpContext& ctx, const TBlob& in, const TBlob& out,
                    const dmlc::optional<int>& n, const int batch_size) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK(!n.has_value() || n.value() > 0);
  CHECK_GE(batch_size, 1);
  if (out.Size() == 0) return;

  int n_ffts = in.shape_.ProdShape(0, in.ndim() - 1);
  int len_fft = out.shape_[in.ndim() - 1];

  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_COMPLEX(in.type_flag_, IType, {
    MSHADOW_COMPLEX_TYPE_SWITCH(out.type_flag_, OType, {
      Tensor<xpu, 2, IType> in_data = in.get_with_shape<xpu, 2, IType>(
          Shape2(n_ffts, in.shape_[in.ndim() - 1]), s);
      Tensor<xpu, 2, OType> out_data = out.get_with_shape<xpu, 2, OType>(
          Shape2(n_ffts, len_fft), s);
      FFTExec<xpu, IType, OType, OType>(ctx, in_data, out_data, OType(0),
                                        n_ffts, len_fft, batch_size);
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
void FFTBackwardImpl(const OpContext& ctx, const TBlob& ograd,
                     const TBlob& igrad, const dmlc::optional<int>& n,
                     const int batch_size) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK(!n.has_value() || n.value() > 0);
  CHECK_GE(batch_size, 1);
  if (igrad.Size() == 0) return;

  int n_ffts = ograd.shape_.ProdShape(0, ograd.ndim() - 1);
  int len_fft = ograd.shape_[ograd.ndim() - 1];
  int grad_dim = igrad.shape_[ograd.ndim() - 1];

  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_COMPLEX_TYPE_SWITCH(ograd.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH_WITH_COMPLEX(igrad.type_flag_, OType, {
      Tensor<xpu, 2, IType> in_data =
          ograd.get_with_shape<xpu, 2, IType>(Shape2(n_ffts, len_fft), s);
      Tensor<xpu, 2, OType> out_data =
          igrad.get_with_shape<xpu, 2, OType>(Shape2(n_ffts, grad_dim), s);
      FFTExec<xpu, IType, OType, IType>(ctx, in_data, out_data, IType(0),
                                        n_ffts, len_fft, batch_size, grad_dim);
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
