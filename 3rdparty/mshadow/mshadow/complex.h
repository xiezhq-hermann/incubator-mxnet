/*!
 *  Copyright (c) 2019 by Contributors
 * \file complex.h
 * \brief definition of complex (complex64 and complex128) type.
 *
 * \author Zhiqiang Xie
 */
#ifndef MSHADOW_COMPLEX_H_
#define MSHADOW_COMPLEX_H_
#include "./base.h"

#if (MSHADOW_USE_CUDA)
  #define MSHADOW_CUDA_COMPLEX 1
  #include <cuComplex.h>
#else
  #define MSHADOW_CUDA_COMPLEX 0
#endif

/*! \brief namespace for mshadow */
namespace mshadow {
/* \brief name space for host/device portable complex value */
namespace complex {
#define MSHADOW_COMPLEX_OPERATOR(RTYPE, OP)                               \
  template<typename T>                                                    \
  MSHADOW_XINLINE RTYPE operator OP (RTYPE a, T b) {                      \
    return RTYPE(a OP RTYPE(b));  /* NOLINT(*) */                         \
  }                                                                       \
  template<typename T>                                                    \
  MSHADOW_XINLINE RTYPE operator OP (T a, RTYPE b) {                      \
    return RTYPE(RTYPE(a) OP b);  /* NOLINT(*) */                         \
  }

#define MSHADOW_COMPLEX_ASSIGNOP(AOP, OP)                                        \
  template<typename T>                                                           \
  MSHADOW_XINLINE complex64 operator AOP (const T& a) {                          \
    return *this = complex64(complex64(*this) OP complex64(a));  /* NOLINT(*)*/  \
  }                                                                              \
  template<typename T>                                                           \
  MSHADOW_XINLINE complex64 operator AOP (const volatile T& a) volatile {        \
    return *this = complex64(complex64(*this) OP complex64(a));  /* NOLINT(*)*/  \
  }

#define MSHADOW_COMPLEX_ASSIGNOP_DOUBLE(AOP, OP)                                    \
  template<typename T>                                                              \
  MSHADOW_XINLINE complex128 operator AOP (const T& a) {                            \
    return *this = complex128(complex128(*this) OP complex128(a));  /* NOLINT(*)*/  \
  }                                                                                 \
  template<typename T>                                                              \
  MSHADOW_XINLINE complex128 operator AOP (const volatile T& a) volatile {          \
    return *this = complex128(complex128(*this) OP complex128(a));  /* NOLINT(*)*/  \
  }

class complex128;

class MSHADOW_ALIGNED(8) complex64 {
 public:
  union {
    std::complex<float> complex64_;
#if MSHADOW_CUDA_COMPLEX
    cuFloatComplex cucomplex64_;
#endif  // MSHADOW_CUDA_COMPLEX
  };

  MSHADOW_XINLINE complex64() {}
  MSHADOW_XINLINE complex64(const float& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const double& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const int8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const uint8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const int32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const uint32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const int64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const uint64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex64(const std::complex<float>& value ) {
    complex64_ = value;
  }
  MSHADOW_XINLINE explicit complex64(const std::complex<double>& value ) {
    complex64_ = std::complex<float>(value);
  }

#if MSHADOW_CUDA_COMPLEX
  MSHADOW_XINLINE explicit complex64(const cuFloatComplex& value) {
    cucomplex64_ = value;
  }
  MSHADOW_XINLINE explicit complex64(const cuDoubleComplex& value) {
    cucomplex64_ = cuComplexDoubleToFloat(value);
  }
#endif  // MSHADOW_CUDA_COMPLEX

#if MSHADOW_CUDA_COMPLEX
  complex64 operator+(complex64 a){
    return complex64(cuCaddf(cucomplex64_, a.cucomplex64_));
  }
  complex64 operator-(complex64 a){
    return complex64(cuCsubf(cucomplex64_, a.cucomplex64_));
  }
  complex64 operator*(complex64 a){
    return complex64(cuCmulf(cucomplex64_, a.cucomplex64_));
  }
  complex64 operator/(complex64 a){
    return complex64(cuCdivf(cucomplex64_, a.cucomplex64_));
  }
  // operator complex128() const{
  //   return complex128(cuComplexFloatToDouble(cucomplex64_));
  // }
#else
  complex64 operator+(complex64 a){
    return complex64(complex64_ + a.complex64_);
  }
  complex64 operator-(complex64 a){
    return complex64(complex64_ - a.complex64_);
  }
  complex64 operator*(complex64 a){
    return complex64(complex64_ * a.complex64_);
  }
  complex64 operator/(complex64 a){
    return complex64(complex64_ / a.complex64_);
  }
  // operator complex128() const{
  //   return complex128(std::complex<double>(complex64_));
  // }
#endif

  MSHADOW_COMPLEX_ASSIGNOP(+=, +)
  MSHADOW_COMPLEX_ASSIGNOP(-=, -)
  MSHADOW_COMPLEX_ASSIGNOP(*=, *)
  MSHADOW_COMPLEX_ASSIGNOP(/=, /)

 private:
  template<typename T>
  MSHADOW_XINLINE void constructor(const T& value) {
#if (MSHADOW_CUDA_COMPLEX && defined(__CUDA_ARCH__))
    cucomplex64_ = make_cuFloatComplex(float(value), 0);  // NOLINT(*)
#else
    complex64_ = std::complex<float>(float(value), 0);  // NOLINT(*)
#endif
  } 
};

class MSHADOW_ALIGNED(16) complex128 {
 public:
  union {
    std::complex<double> complex128_;
#if MSHADOW_CUDA_COMPLEX
    cuDoubleComplex cucomplex128_;
#endif  // MSHADOW_CUDA_COMPLEX
  };

  MSHADOW_XINLINE complex128() {}
  MSHADOW_XINLINE complex128(const float& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const double& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const int8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const uint8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const int32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const uint32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const int64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const uint64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const std::complex<double>& value ) {
    complex128_ = value;
  }
  MSHADOW_XINLINE explicit complex128(const std::complex<float>& value ) {
    complex128_ = std::complex<double>(value);
  }

#if MSHADOW_CUDA_COMPLEX
  MSHADOW_XINLINE explicit complex128(const cuDoubleComplex& value) {
    cucomplex128_ = value;
  }
  MSHADOW_XINLINE explicit complex128(const cuFloatComplex& value) {
    cucomplex128_ = cuComplexFloatToDouble(value);
  }
#endif  // MSHADOW_CUDA_COMPLEX

#if MSHADOW_CUDA_COMPLEX
  complex128 operator+(complex128 a){
    return complex128(cuCadd(cucomplex128_, a.cucomplex128_));
  }
  complex128 operator-(complex128 a){
    return complex128(cuCsub(cucomplex128_, a.cucomplex128_));
  }
  complex128 operator*(complex128 a){
    return complex128(cuCmul(cucomplex128_, a.cucomplex128_));
  }
  complex128 operator/(complex128 a){
    return complex128(cuCdiv(cucomplex128_, a.cucomplex128_));
  }
  // operator complex64() const{
  //   return complex64(cuComplexDoubleToFloat(cucomplex128_));
  // }
#else
  complex128 operator+(complex128 a){
    return complex128(complex128_ + a.complex128_);
  }
  complex128 operator-(complex128 a){
    return complex128(complex128_ - a.complex128_);
  }
  complex128 operator*(complex128 a){
    return complex128(complex128_ * a.complex128_);
  }
  complex128 operator/(complex128 a){
    return complex128(complex128_ / a.complex128_);
  }
  // operator complex64() const{
  //   return complex64(std::complex<float>(complex128_));
  // }
#endif

  MSHADOW_COMPLEX_ASSIGNOP_DOUBLE(+=, +)
  MSHADOW_COMPLEX_ASSIGNOP_DOUBLE(-=, -)
  MSHADOW_COMPLEX_ASSIGNOP_DOUBLE(*=, *)
  MSHADOW_COMPLEX_ASSIGNOP_DOUBLE(/=, /)

 private:
  template<typename T>
  MSHADOW_XINLINE void constructor(const T& value) {
#if (MSHADOW_CUDA_COMPLEX && defined(__CUDA_ARCH__))
    cucomplex128_ = make_cuDoubleComplex(double(value), 0);  // NOLINT(*)
#else
    complex128_ = std::complex<double>(double(value), 0);  // NOLINT(*)
#endif
  }
};

/*! \brief overloaded + operator for complex64 */
MSHADOW_COMPLEX_OPERATOR(complex64, +)
/*! \brief overloaded - operator for complex64 */
MSHADOW_COMPLEX_OPERATOR(complex64, -)
/*! \brief overloaded * operator for complex64 */
MSHADOW_COMPLEX_OPERATOR(complex64, *)
/*! \brief overloaded / operator for complex64 */
MSHADOW_COMPLEX_OPERATOR(complex64, /)

/*! \brief overloaded + operator for complex128 */
MSHADOW_COMPLEX_OPERATOR(complex128, +)
/*! \brief overloaded - operator for complex128 */
MSHADOW_COMPLEX_OPERATOR(complex128, -)
/*! \brief overloaded * operator for complex128 */
MSHADOW_COMPLEX_OPERATOR(complex128, *)
/*! \brief overloaded / operator for complex128 */
MSHADOW_COMPLEX_OPERATOR(complex128, /)

}  // namespace complex
}  // namespace mshadow
#endif  // MSHADOW_COMPLEX_H_
