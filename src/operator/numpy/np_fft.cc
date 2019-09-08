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
 * Copyright (c) 2015 by Contributors
 * \file fft-inl.h
 * \brief
 * \author Chen Zhu
*/
#include "./np_fft-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateNPOp<cpu>(NPFFTParam param, int dtype) {
  LOG(FATAL) << "fft is only available for GPU.";
  return nullptr;
}

Operator *NPFFTProp::CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                                                    std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateNPOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(NPFFTParam);

MXNET_REGISTER_OP_PROPERTY(_np_fft, NPFFTProp)
.describe(R"code(Apply 1D FFT to input"

.. note:: `fft` is only available on GPU.

Currently support native `complex64` data type for both input and output, which share the same size

Example::

   data_cpu = np.random.normal(0,1,(3,4)).astype(np.complex64)
   data_cpu.imag = np.random.normal(0,1,(3,4))
   data_gpu = mx.nd.array(data_cpu, dtype="complex64", ctx=mx.gpu(0))
   out = mx.contrib.ndarray.fft(data_gpu)

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the FFTOp.")
.add_arguments(NPFFTParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
