#ifndef _SYRAH_FIXED_VECTOR_H_
#define _SYRAH_FIXED_VECTOR_H_

#include <algorithm>
#include <float.h>
#include <iostream>
#include <typeinfo>
#include <stdint.h>
#include "Preprocessor.h"

#ifdef __ARM_NEON__
#define SYRAH_USE_NEON 1
#define SYRAH_USE_AVX  0
#define SYRAH_USE_SSE  0
#define SYRAH_USE_LRB  0
#elif defined(__AVX__)
#define SYRAH_USE_NEON 0
#define SYRAH_USE_AVX  1
#define SYRAH_USE_SSE  0
#define SYRAH_USE_LRB  0
#elif defined(__SSE3__)
#define SYRAH_USE_NEON 0
#define SYRAH_USE_SSE  1
#define SYRAH_USE_LRB  0
#define SYRAH_USE_AVX  0
#endif

#if SYRAH_USE_SSE || SYRAH_USE_NEON
#define SYRAH_SIMD_BYTE_WIDTH 16
#define SYRAH_MASK_BYTE_WIDTH 4
#elif SYRAH_USE_AVX
#define SYRAH_SIMD_BYTE_WIDTH 32
#define SYRAH_MASK_BYTE_WIDTH 4
#elif SYRAH_USE_LRB
#define SYRAH_SIMD_BYTE_WIDTH 64
#define SYRAH_MASK_BYTE_WIDTH 4
#else
#error "UNKNOWN VECTOR OPTION"
#endif

#define SYRAH_IS_MULTIPLE(N, M) (((N) % (M)) == 0)
#define SYRAH_IS_SIMD_MULTIPLE_WIDTH(N, BYTE_WIDTH) (SYRAH_IS_MULTIPLE(N, SYRAH_SIMD_BYTE_WIDTH))
#define SYRAH_IS_SIMD_MULTIPLE(N, TYPE) (SYRAH_IS_MULTIPLE((N) * sizeof(TYPE), SYRAH_SIMD_BYTE_WIDTH))
#define SYRAH_IS_SIMD_MULTIPLE_MASK(N) (SYRAH_IS_MULTIPLE((N) * SYRAH_MASK_BYTE_WIDTH, SYRAH_SIMD_BYTE_WIDTH))

// SSE is 16 bytes (4 floats, 2 doubles, etc)


namespace syrah {
  template<int N, bool SIMDMultiple = SYRAH_IS_SIMD_MULTIPLE_MASK(N)>
  class FixedVectorMask {
  public:
    FixedVectorMask(bool value) {
      for (int i = 0; i < N; i++) {
        data[i] = value;
      }
    }
    FixedVectorMask() {}
    FixedVectorMask(const bool* values) {
      for (int i = 0; i < N; i++) {
        data[i] = values[i];
      }
    }

    bool get(int i) const {
      return data[i];
    }
    void set(int i, bool val) {
      data[i] = val;
    }

    FixedVectorMask operator|(const FixedVectorMask& v) const {
      FixedVectorMask<N> result;
      for (int i = 0; i < N; i++) {
        result.data[i] = (data[i] | v.data[i]);
      }
      return result;
    }

    FixedVectorMask& operator|=(const FixedVectorMask& v) {
      for (int i = 0; i < N; i++) {
        data[i] |= v[i];
      }
      return *this;
    }

    FixedVectorMask operator&(const FixedVectorMask& v) const {
      FixedVectorMask result;
      for (int i = 0; i < N; i++) {
        result.data[i] = (data[i] & v[i]);
      }
      return result;
    }

    FixedVectorMask operator&=(const FixedVectorMask& v) {
      for (int i = 0; i < N; i++) {
        data[i] &= v[i];
      }
      return *this;
    }

    FixedVectorMask operator^(const FixedVectorMask& v) const {
      FixedVectorMask result;
      for (int i = 0; i < N; i++) {
        result.data[i] = (data[i] ^ v[i]);
      }
      return result;
    }

    FixedVectorMask operator^=(const FixedVectorMask& v) {
      for (int i = 0; i < N; i++) {
        data[i] ^= v[i];
      }
      return *this;
    }

    FixedVectorMask operator!() const {
      FixedVectorMask result;
      for (int i = 0; i < N; i++) {
        result.data[i] = (!data[i]);
      }
      return result;
    }

    // Simple scalar implementation, an specialized implementation
    // might use bit vectors or sse masks or LRB mask vectors.
    bool data[N];
  };

  template<int N>
  SYRAH_FORCEINLINE bool All(const FixedVectorMask<N>& v) {
    for (int i = 0; i < N; i++) {
      if (!v.get(i)) return false;
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE bool Any(const FixedVectorMask<N>& v) {
    for (int i = 0; i < N; i++) {
      if (v.get(i)) return true;
    }
    return false;
  }

  template<int N>
  SYRAH_FORCEINLINE bool None(const FixedVectorMask<N>& v) {
    for (int i = 0; i < N; i++) {
      if (v[i].get(i)) return false;
    }
    return true;
  }

  // v1 & (!v2)
  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N> andNot(const FixedVectorMask<N>& v1, const FixedVectorMask<N>& v2) {
    return v1 & (!v2);
  }

  template<int N>
  SYRAH_FORCEINLINE std::ostream& operator<<(std::ostream& os, const FixedVectorMask<N>& v) {
    for (int i = 0; i < N; i++) {
      os << v.get(i) << " ";
    }
    return os;
  }

  template<typename ElemType, int N, bool SIMDMultiple = SYRAH_IS_SIMD_MULTIPLE(N, ElemType)>
  class FixedVector {
  public:
    FixedVector(const ElemType value) {
      load(value);
    }
    FixedVector() {}

    FixedVector(const ElemType* values) {
      load(values);
    }

    FixedVector(const ElemType* values, bool aligned) {
      load_aligned(values);
    }

    FixedVector(const ElemType* values, const FixedVectorMask<N>& mask, const ElemType default_value) {
      load(values, mask, default_value);
    }

    FixedVector(const ElemType* values, const FixedVectorMask<N>& mask, const ElemType default_value, bool aligned) {
      load_aligned(values, mask, default_value);
    }

    FixedVector(const ElemType* base, const FixedVector<int, N>& offsets, const int scale) {
      gather(base, offsets, scale);
    }

    FixedVector(const ElemType* base, const FixedVector<int, N>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      gather(base, offsets, scale, mask);
    }

    FixedVector(const FixedVector& v) {
      for (int i = 0; i < N; i++) {
        data[i] = v[i];
      }
    }

    FixedVector& operator=(const FixedVector& v) {
      for (int i = 0; i < N; i++) {
        data[i] = v.data[i];
      }
    }

    template<typename OtherType>
    explicit FixedVector(const FixedVector<OtherType, N>& v) {
      for (int i = 0; i < N; i++) {
        data[i] = static_cast<ElemType>(v[i]);
      }
    }

    template<typename OtherType>
    static FixedVector reinterpret(const FixedVector<OtherType, N>& v) {
      struct UnionType {
        union {
          ElemType el;
          OtherType other;
        };
      };
      FixedVector result;
      const UnionType* union_other = reinterpret_cast<const UnionType*>(v.data);
      for (int i = 0; i < N; i++) {
        result.data[i] = union_other[i].el;
      }
      return result;
    }

    static FixedVector Zero() {
      FixedVector result;
      for (int i = 0; i < N; i++) { result.data[i] = ElemType(0); }
      return result;
    }

    const ElemType& operator[](int i) const {
      return data[i];
    }
    ElemType& operator[](int i) {
      return data[i];
    }

    void load(const ElemType value) {
      for (int i = 0; i < N; i++) {
        data[i] = static_cast<ElemType>(value);
      }
    }

    void load(const ElemType* values) {
      for (int i = 0; i < N; i++) {
        data[i] = static_cast<ElemType>(values[i]);
      }
    }

    void load_aligned(const ElemType* values) {
      load(values);
    }

    void load(const ElemType* values, const FixedVectorMask<N>& mask, const ElemType default_value) {
      for (int i = 0; i < N; i++) {
        if (mask[i]) {
          data[i] = static_cast<ElemType>(values[i]);
        } else {
          data[i] = default_value;
        }
      }
    }

    void load_aligned(const ElemType* values, const FixedVectorMask<N>& mask, const ElemType default_value) {
      load(values, mask, default_value);
    }

    void gather(const ElemType* base, const FixedVector<int, N>& offsets, const int scale) {
      for (int i = 0; i < N; i++) {
        const ElemType* addr = reinterpret_cast<const ElemType*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
        data[i] = *addr;
      }
    }

    void gather(const ElemType* base, const FixedVector<int, N>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          const ElemType* addr = reinterpret_cast<const ElemType*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
          data[i] = *addr;
        }
      }
    }

    void store(ElemType* dst) const {
      for (int i = 0; i < N; i++) {
        dst[i] = data[i];
      }
    }

    void store_aligned(ElemType* dst) const {
      store(dst);
    }

    void store_aligned_stream(ElemType* dst) const {
      store(dst);
    }

    void store(ElemType* dst, const FixedVectorMask<N>& mask) const {
      for (int i = 0; i < N; i++) {
        if (mask.get(i))
          dst[i] = data[i];
      }
    }

    void store_aligned(ElemType* dst, const FixedVectorMask<N>& mask) const {
      store(dst, mask);
    }

    void scatter(ElemType* base, const FixedVector<int, N>& offsets, const int scale) const {
      for (int i = 0; i < N; i++) {
        ElemType* addr = reinterpret_cast<ElemType*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
        *addr = data[i];
      }
    }

    void scatter(ElemType* base, const FixedVector<int, N>& offsets, const int scale, const FixedVectorMask<N>& mask) const {
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          ElemType* addr = reinterpret_cast<ElemType*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
          *addr = data[i];
        }
      }
    }

    // merge v into our vector based on mask. Could also call this
    // update, but merge seems better...
    void merge(const FixedVector& v, const FixedVectorMask<N>& mask) {
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) data[i] = v[i];
      }
    }

#define SYRAH_BINARY_OP(RES_TYPE, OP) \
SYRAH_FORCEINLINE RES_TYPE operator OP(const FixedVector& v) const { \
  RES_TYPE result; \
  for (int i = 0; i < N; i++) { \
    result.data[i] = data[i] OP v[i]; \
  } \
  return result; \
}

    SYRAH_BINARY_OP(FixedVector, +)
    SYRAH_BINARY_OP(FixedVector, -)
    SYRAH_BINARY_OP(FixedVector, *)
    SYRAH_BINARY_OP(FixedVector, /)
    SYRAH_BINARY_OP(FixedVector, %)
    SYRAH_BINARY_OP(FixedVector, &)
    SYRAH_BINARY_OP(FixedVector, |)
    SYRAH_BINARY_OP(FixedVector, ^)
    SYRAH_BINARY_OP(FixedVector, <<)
    SYRAH_BINARY_OP(FixedVector, >>)

#undef SYRAH_BINARY_OP

#define SYRAH_MASK_OP(RES_TYPE, OP) \
SYRAH_FORCEINLINE RES_TYPE operator OP(const FixedVector& v) const { \
  RES_TYPE result; \
  for (int i = 0; i < N; i++) { \
    result.set(i, data[i] OP v[i]);             \
  } \
  return result; \
}

    SYRAH_MASK_OP(FixedVectorMask<N>, <)
    SYRAH_MASK_OP(FixedVectorMask<N>, <=)
    SYRAH_MASK_OP(FixedVectorMask<N>, ==)
    SYRAH_MASK_OP(FixedVectorMask<N>, >=)
    SYRAH_MASK_OP(FixedVectorMask<N>, >)
    SYRAH_MASK_OP(FixedVectorMask<N>, !=)

#undef SYRAH_MASK_OP


#define SYRAH_BINARY_UPDATE(OP) \
SYRAH_FORCEINLINE FixedVector& operator OP(const FixedVector& v) { \
  for (int i = 0; i < N; i++) { \
    data[i] OP v[i]; \
  } \
  return *this; \
}

    SYRAH_BINARY_UPDATE(+=)
    SYRAH_BINARY_UPDATE(-=)
    SYRAH_BINARY_UPDATE(*=)
    SYRAH_BINARY_UPDATE(/=)
    SYRAH_BINARY_UPDATE(%=)
    SYRAH_BINARY_UPDATE(&=)
    SYRAH_BINARY_UPDATE(|=)
    SYRAH_BINARY_UPDATE(^=)
    SYRAH_BINARY_UPDATE(>>=)
    SYRAH_BINARY_UPDATE(<<=)

    ElemType MaxElement() const {
      ElemType result = data[0];
      for (int i = 1; i < N; i++) {
        result = std::max(result, data[i]);
      }
      return result;
    }

    ElemType MinElement() const {
      ElemType result = data[0];
      for (int i = 1; i < N; i++) {
        result = std::min(result, data[i]);
      }
      return result;
    }

    ElemType foldMin() const { return MinElement(); }
    ElemType foldMax() const { return MaxElement(); }
    ElemType foldSum() const {
      ElemType result = ElemType(0);
      for (int i = 0; i < N; i++) {
        result += data[i];
      }
      return result;
    }
    ElemType foldProd() const {
      ElemType result = ElemType(1);
      for (int i = 0; i < N; i++) {
        result *= data[i];
      }
      return result;
    }

    ElemType data[N];
  };

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator+(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = scalar + vector[i];
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator+(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = vector[i] + scalar;
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator-(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = scalar - vector[i];
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator-(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = vector[i] - scalar;
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator*(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = scalar * vector[i];
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator*(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = vector[i] * scalar;
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator/(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = scalar / vector[i];
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> operator/(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = vector[i] / scalar;
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator<(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, scalar < vector[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator<(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, vector[i] < scalar);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator<=(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, scalar <= vector[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator<=(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, vector[i] <= scalar);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator==(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, scalar == vector[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator==(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, vector[i] == scalar);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator>=(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, scalar >= vector[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator>=(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, vector[i] >= scalar);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator>(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, scalar > vector[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator>(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, vector[i] > scalar);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator!=(const ElemType& scalar, const FixedVector<ElemType, N, SIMDMultiple>& vector) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, scalar != vector[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVectorMask<N, SIMDMultiple> operator!=(const FixedVector<ElemType, N, SIMDMultiple>& vector, const ElemType& scalar) {
    FixedVectorMask<N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result.set(i, vector[i] != scalar);
    }
    return result;
  }


  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> min(const FixedVector<ElemType, N, SIMDMultiple>& v1,
                                                               const FixedVector<ElemType, N, SIMDMultiple>& v2) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = std::min(v1[i], v2[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> max(const FixedVector<ElemType, N, SIMDMultiple>& v1,
                                                               const FixedVector<ElemType, N, SIMDMultiple>& v2) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = std::max(v1[i], v2[i]);
    }
    return result;
  }

  // a * b + c
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> madd(const FixedVector<ElemType, N, SIMDMultiple>& a,
                                                     const FixedVector<ElemType, N, SIMDMultiple>& b,
                                                     const FixedVector<ElemType, N, SIMDMultiple>& c) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = a[i] * b[i] + c[i];
    }
    return result;
  }

  // output = (mask[i]) ? a : b
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> select(const FixedVector<ElemType, N, SIMDMultiple>& a,
                                                                  const FixedVector<ElemType, N, SIMDMultiple>& b,
                                                                  const FixedVectorMask<N>& mask) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = (mask.get(i)) ? a[i] : b[i];
    }
    return result;
  }

  // output[i] = a[N - 1 - i] (output is reversed a)
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> reverse(const FixedVector<ElemType, N, SIMDMultiple>& a) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = a[N - 1 - i];
    }
    return result;
  }


  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE std::ostream& operator<<(std::ostream& os, const FixedVector<ElemType, N, SIMDMultiple>& v) {
    for (int i = 0; i < N; i++) {
      os << v[i] << " ";
    }
    return os;
  }

  template<int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE std::ostream& operator<<(std::ostream& os, const FixedVector<uint8_t, N, SIMDMultiple>& v) {
    for (int i = 0; i < N; i++) {
      os << (int)v[i] << " ";
    }
    return os;
  }
};

#include "FixedVectorMath.h"

#if SYRAH_USE_SSE
#include "sse/FixedVector_SSE.h"
#elif SYRAH_USE_AVX
#include "avx/FixedVector_AVX.h"
#elif SYRAH_USE_LRB
#include "lrb/FixedVector_LRB.h"
#elif SYRAH_USE_NEON
#include "neon/FixedVector_NEON.h"
#else
#error "UNKNOWN CHOICE"
#endif

#endif // _SYRAH_FIXED_VECTOR_H_
