#ifndef _SYRAH_FIXED_VECTOR_SSE_CASTS_H_
#define _SYRAH_FIXED_VECTOR_SSE_CASTS_H_

namespace syrah {
#define SYRAH_SSE_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true>::FixedVector(const FixedVector<int, N> &v) {
    SYRAH_SSE_LOOP(i) {
      data[i] = _mm_cvtepi32_ps(v.data[i]);
    }
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> FixedVector<float, N, true>::reinterpret(const FixedVector<int, N, true>& v) {
    FixedVector<float, N> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> FixedVector<int, N, true>::reinterpret(const FixedVector<float, N, true>& v) {
    FixedVector<int, N> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castps_si128(v.data[i]);
    }
    return result;
  }

};

#endif // _SYRAH_FIXED_VECTOR_SSE_CASTS_H_
