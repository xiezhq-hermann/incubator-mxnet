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

inline bool FFTShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape>* in_attrs,
                      std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline int TypeCast(const int& itype) {
  using namespace mshadow;
  switch (itype)
  {
  case kFloat32: case kFloat16: case kUint8: case kInt8: case kInt32: case kInt64: case kComplex64:
    return kComplex64;
  case kFloat64: case kComplex128:
    return kComplex128;
  default:
    LOG(FATAL) << "Unknown type enum " << itype;
  }
  return -1;
}

inline bool FFTType(const nnvm::NodeAttrs& attrs,
                      std::vector<int>* in_attrs,
                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, TypeCast(in_attrs->at(0)));
  // TYPE_ASSIGN_CHECK(*in_attrs, 0, TypeCast(out_attrs->at(0)));

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(NPFFTParam);

NNVM_REGISTER_OP(_np_fft)
.set_attr_parser(ParamParser<NPFFTParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", FFTShape)
.set_attr<nnvm::FInferType>("FInferType", FFTType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
// .set_attr<FCompute>("FCompute<cpu>", FFTForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
                            ElemwiseGradUseNone{"_backward_np_fft"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NPFFTParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_fft)
.set_attr_parser(ParamParser<NPFFTParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });
// .set_attr<FCompute>("FCompute<cpu>", FFTBackward<cpu>);

}  // namespace op
}  // namespace mxnet
