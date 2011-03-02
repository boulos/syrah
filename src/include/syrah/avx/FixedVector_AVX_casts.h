#ifndef _SYRAH_FIXED_VECTOR_AVX_CASTS_H_
#define _SYRAH_FIXED_VECTOR_AVX_CASTS_H_

namespace syrah {

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true>::FixedVector(const FixedVector<int, N> &v) {
    SYRAH_UNROLL(N/8)
      for (int i = 0; i < N/8; i++) {
        __m128 cvt0 = _mm_cvtepi32_ps(v.data[2*i + 0]);
        __m128 cvt1 = _mm_cvtepi32_ps(v.data[2*i + 1]);
        data[i] = _mm256_insertf128_ps(_mm256_castps128_ps256(cvt0), cvt1, 1);
      }
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> FixedVector<float, N, true>::reinterpret(const FixedVector<int, N, true> &v) {
    FixedVector<float, N> result;
    SYRAH_UNROLL(N/8)
      for (int i = 0; i < N/8; i++) {
        __m128 cvt0 = _mm_castsi128_ps(v.data[2*i + 0]);
        __m128 cvt1 = _mm_castsi128_ps(v.data[2*i + 1]);
        result.data[i] = _mm256_insertf128_ps(_mm256_castps128_ps256(cvt0), cvt1, 1);
      }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> FixedVector<int, N, true>::reinterpret(const FixedVector<float, N, true> &v) {
    FixedVector<int, N> result;
    SYRAH_UNROLL(N/8)
      for (int i = 0; i < N/8; i++) {
        __m128 lo_half = _mm256_extractf128_ps(v.data[i], 0);
        __m128 hi_half = _mm256_extractf128_ps(v.data[i], 1);
        __m128i cvt0 = _mm_castps_si128(lo_half);
        __m128i cvt1 = _mm_castps_si128(hi_half);
        result.data[2*i + 0] = cvt0;
        result.data[2*i + 1] = cvt1;
      }
    return result;
  }

};

#endif // _SYRAH_FIXED_VECTOR_AVX_CASTS_H_
