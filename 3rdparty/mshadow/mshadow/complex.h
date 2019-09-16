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
#define MSHADOW_COMPLEX_OPERATOR(DTYPE, OP)                               \
  template<typename T, typename = typename\
  std::enable_if<!(std::is_same<T, DTYPE>::value)>::type>  \
  MSHADOW_XINLINE DTYPE operator OP (const DTYPE& a, const T& b) {        \
    return a OP DTYPE(b);  /* NOLINT(*) */                                \
  }                                                                       \
  template<typename T, typename = typename\
  std::enable_if<!(std::is_same<T, DTYPE>::value)>::type>  \
  MSHADOW_XINLINE DTYPE operator OP (const T& a, const DTYPE& b) {        \
    return DTYPE(a) OP b;  /* NOLINT(*) */                                \
  }                                                                       \

#define MSHADOW_COMPLEX_ASSIGNOP(DTYPE, AOP, OP)                          \
  template<typename T>                                                    \
  MSHADOW_XINLINE DTYPE operator AOP (const T& a) {                       \
    return *this = DTYPE(*this OP DTYPE(a));  /* NOLINT(*)*/              \
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
  MSHADOW_XINLINE explicit complex64(const complex128& value);

  MSHADOW_XINLINE operator float() const {
#if MSHADOW_CUDA_COMPLEX
    return cuCrealf(cucomplex64_);
#else
    return complex64_.real();
#endif
  }

#if MSHADOW_CUDA_COMPLEX
  MSHADOW_XINLINE explicit complex64(const cuFloatComplex& value) {
    cucomplex64_ = value;
  }
  MSHADOW_XINLINE explicit complex64(const cuDoubleComplex& value) {
    cucomplex64_ = cuComplexDoubleToFloat(value);
  }
#else
  MSHADOW_XINLINE explicit complex64(const std::complex<float>& value ) {
    complex64_ = value;
  }
  MSHADOW_XINLINE explicit complex64(const std::complex<double>& value ) {
    complex64_ = std::complex<float>(value);
  }
#endif  // MSHADOW_CUDA_COMPLEX

#if MSHADOW_CUDA_COMPLEX
  complex64 operator+(const complex64& a){
    return complex64(cuCaddf(cucomplex64_, a.cucomplex64_));
  }
  complex64 operator-(const complex64& a){
    return complex64(cuCsubf(cucomplex64_, a.cucomplex64_));
  }
  complex64 operator*(const complex64& a){
    return complex64(cuCmulf(cucomplex64_, a.cucomplex64_));
  }
  complex64 operator/(const complex64& a){
    return complex64(cuCdivf(cucomplex64_, a.cucomplex64_));
  }
#else
  complex64 operator+(const complex64& a){
    return complex64(complex64_ + a.complex64_);
  }
  complex64 operator-(const complex64& a){
    return complex64(complex64_ - a.complex64_);
  }
  complex64 operator*(const complex64& a){
    return complex64(complex64_ * a.complex64_);
  }
  complex64 operator/(const complex64& a){
    return complex64(complex64_ / a.complex64_);
  }
#endif

  MSHADOW_XINLINE complex64 operator+() {
    return *this;
  }

  MSHADOW_XINLINE complex64 operator-() {
#if MSHADOW_CUDA_COMPLEX
    return complex64(make_cuFloatComplex(-cuCrealf(cucomplex64_), -cuCimagf(cucomplex64_)));
#else
    return -(*this);
#endif
  }

  template<typename T>
  MSHADOW_XINLINE complex64 operator=(const T& a) {
    return *this = complex64(a);  /* NOLINT(*)*/
  }

  MSHADOW_XINLINE complex64 operator=(const complex64& a) {
  #if MSHADOW_CUDA_COMPLEX
    cucomplex64_ = a.cucomplex64_;
  #else
    complex64_ = a.complex64_;
  #endif
    return a;
  }

  MSHADOW_COMPLEX_ASSIGNOP(complex64, +=, +)
  MSHADOW_COMPLEX_ASSIGNOP(complex64, -=, -)
  MSHADOW_COMPLEX_ASSIGNOP(complex64, *=, *)
  MSHADOW_COMPLEX_ASSIGNOP(complex64, /=, /)

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
  MSHADOW_XINLINE complex128(const double& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const float& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const int8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const uint8_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const int32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const uint32_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const int64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const uint64_t& value) { constructor(value); }
  MSHADOW_XINLINE explicit complex128(const complex64& value);

  MSHADOW_XINLINE operator double() const {
#if MSHADOW_CUDA_COMPLEX
    return cuCreal(cucomplex128_);
#else
    return complex128_.real();
#endif
  }

#if MSHADOW_CUDA_COMPLEX
  MSHADOW_XINLINE explicit complex128(const cuDoubleComplex& value) {
    cucomplex128_ = value;
  }
  MSHADOW_XINLINE explicit complex128(const cuFloatComplex& value) {
    cucomplex128_ = cuComplexFloatToDouble(value);
  }
#else
  MSHADOW_XINLINE explicit complex128(const std::complex<double>& value ) {
    complex128_ = value;
  }
  MSHADOW_XINLINE explicit complex128(const std::complex<float>& value ) {
    complex128_ = std::complex<double>(value);
  }
#endif  // MSHADOW_CUDA_COMPLEX

#if MSHADOW_CUDA_COMPLEX
  complex128 operator+(const complex128& a){
    return complex128(cuCadd(cucomplex128_, a.cucomplex128_));
  }
  complex128 operator-(const complex128& a){
    return complex128(cuCsub(cucomplex128_, a.cucomplex128_));
  }
  complex128 operator*(const complex128& a){
    return complex128(cuCmul(cucomplex128_, a.cucomplex128_));
  }
  complex128 operator/(const complex128& a){
    return complex128(cuCdiv(cucomplex128_, a.cucomplex128_));
  }
#else
  complex128 operator+(const complex128& a){
    return complex128(complex128_ + a.complex128_);
  }
  complex128 operator-(const complex128& a){
    return complex128(complex128_ - a.complex128_);
  }
  complex128 operator*(const complex128& a){
    return complex128(complex128_ * a.complex128_);
  }
  complex128 operator/(const complex128& a){
    return complex128(complex128_ / a.complex128_);
  }
#endif

  MSHADOW_XINLINE complex128 operator+() {
    return *this;
  }

  MSHADOW_XINLINE complex128 operator-() {
#if MSHADOW_CUDA_COMPLEX
    return complex128(make_cuDoubleComplex(-cuCreal(cucomplex128_), -cuCimag(cucomplex128_)));
#else
    return -(*this);
#endif
  }

  template<typename T>
  MSHADOW_XINLINE complex128 operator=(const T& a) {
    return *this = complex128(a);  /* NOLINT(*)*/
  }

  MSHADOW_XINLINE complex128 operator=(const complex128& a) {
  #if MSHADOW_CUDA_COMPLEX
    cucomplex128_ = a.cucomplex128_;
  #else
    complex128_ = a.complex128_;
  #endif
    return a;
  }  

  MSHADOW_COMPLEX_ASSIGNOP(complex128, +=, +)
  MSHADOW_COMPLEX_ASSIGNOP(complex128, -=, -)
  MSHADOW_COMPLEX_ASSIGNOP(complex128, *=, *)
  MSHADOW_COMPLEX_ASSIGNOP(complex128, /=, /)

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

#if MSHADOW_CUDA_COMPLEX
  MSHADOW_XINLINE complex64::complex64(const complex128& value) {
    cucomplex64_ = cuComplexDoubleToFloat(value.cucomplex128_);
  }
  MSHADOW_XINLINE complex128::complex128(const complex64& value) {
    cucomplex128_ = cuComplexFloatToDouble(value.cucomplex64_);
  }
#else
  MSHADOW_XINLINE complex64::complex64(const complex128& value ) {
    complex64_ = std::complex<float>(value.complex128_);
  }
  MSHADOW_XINLINE complex128::complex128(const complex64& value ) {
    complex128_ = std::complex<double>(value.complex64_);
  }
#endif

}  // namespace complex
}  // namespace mshadow
#endif  // MSHADOW_COMPLEX_H_
