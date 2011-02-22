#ifndef _SYRAH_FIXED_VECTOR_LRB_CASTS_H_
#define _SYRAH_FIXED_VECTOR_LRB_CASTS_H_

namespace syrah {
#define SYRAH_LRB_LOOP(index) SYRAH_UNROLL(N/16) \
for (int index = 0; index < N/16; index++)

  // float to int
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true>::FixedVector(const FixedVector<float, N, true>& v) {
    SYRAH_LRB_LOOP(i) {
       data[i] = _mm512_cvt_ps2pi(v.data[i], _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
    }
  }

  // int = reinterpret_cast<int>(float)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> FixedVector<int, N, true>::reinterpret(const FixedVector<float, N, true>& v) {
    FixedVector<int, N, true> result;
    SYRAH_LRB_LOOP(i) {
       result.data[i] = _mm512_castps_si512(v.data[i]);
    }
    return result;
  }

  // int to float
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true>::FixedVector(const FixedVector<int, N> &v) {
    SYRAH_LRB_LOOP(i) {
      data[i] = _mm512_cvt_pi2ps(v.data[i], _MM_EXPADJ_NONE);
    }
  }

  // float = reinterpret_cast<float>(int)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> FixedVector<float, N, true>::reinterpret(const FixedVector<int, N, true>& v) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
       result.data[i] = _mm512_castsi512_ps(v.data[i]);
    }
    return result;
  }
}

#undef SYRAH_LRB_LOOP

#endif // _SYRAH_FIXED_VECTOR_LRB_CASTS_H_
