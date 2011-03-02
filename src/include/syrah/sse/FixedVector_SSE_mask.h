#ifndef _SYRAH_FIXED_VECTOR_SSE_MASK_H_
#define _SYRAH_FIXED_VECTOR_SSE_MASK_H_

namespace syrah {
  // NOTE(boulos): Could instead have this represented as one bit per
  // element and have operator< take _mm_movemask_ps and cast to exact
  // bits each time...
  template<int N>
  class SYRAH_ALIGN(16) FixedVectorMask<N, true> {
  public:
#define SYRAH_SSE_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)

    SYRAH_FORCEINLINE FixedVectorMask(bool value) {
      if (true) {
        __m128 all_on = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
        SYRAH_SSE_LOOP(i) {
          data[i] = all_on;
        }
      } else {
        SYRAH_SSE_LOOP(i) {
          data[i] = _mm_setzero_ps();
        }
      }
    }
    SYRAH_FORCEINLINE FixedVectorMask() {}
    SYRAH_FORCEINLINE FixedVectorMask(const bool* values) {
      const unsigned int kTrue = static_cast<unsigned int>(-1);
      int* int_data = reinterpret_cast<int*>(data);
      for (int i = 0; i < N; i++) {
        int_data[i] = (values[i]) ? kTrue : 0;
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask(const int* values) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_load_ps((float*)&values[4*i]);
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask(const FixedVectorMask<N>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator=(const FixedVectorMask<N>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    // NOTE(boulos): Could also try to do fancy stuff like STL bit
    // vector does to allow operator[].
    SYRAH_FORCEINLINE bool get(int i) const {
      const int* int_data = reinterpret_cast<const int*>(data);
      return int_data[i] == -1;
    }

    SYRAH_FORCEINLINE void set(int i, bool val) {
      int* int_data = reinterpret_cast<int*>(data);
      if (val) {
        int_data[i] = -1;
      } else {
        int_data[i] = 0;
      }
    }

    // NOTE(boulos): I'm intentionally not overloading || as doing so
    // doesn't result in short-circuiting for overloaded ||:
    // http://www.parashift.com/c++-faq-lite/operator-overloading.html#faq-13.9
    // part 19.
    SYRAH_FORCEINLINE FixedVectorMask operator|(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_or_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator|=(const FixedVectorMask& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_or_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator&(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_and_ps(data[i],v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator&=(const FixedVectorMask& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_and_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator^(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_xor_ps(data[i],v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator^=(const FixedVectorMask& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_xor_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE static FixedVectorMask True() {
      // NOTE(boulos): Don't just call return FixedVectorMask(true);
      // to avoid branch.
      FixedVectorMask result;
      __m128 all_on = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
      SYRAH_SSE_LOOP(i) {
        result.data[i] = all_on;
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask False() {
      FixedVectorMask result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_setzero_ps();
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask FirstN(int n) {
      __m128i n_splat = _mm_set1_epi32(n);
      __m128i small_seq = _mm_set_epi32(3, 2, 1, 0);
      __m128i inc_val = _mm_set1_epi32(4);
      FixedVectorMask result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_castsi128_ps(_mm_cmplt_epi32(small_seq, n_splat));
        small_seq = _mm_add_epi32(small_seq, inc_val);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator!() const {
      FixedVectorMask result;
      __m128 all_on = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
      SYRAH_SSE_LOOP(i) {
        // NOTE(boulos): SSE's andnot function is actually (NOT x) & y
        result.data[i] = _mm_andnot_ps(data[i], all_on);
      }
      return result;
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst) const {
      SYRAH_SSE_LOOP(i) {
        _mm_store_ps(&(dst[4*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst) const {
      SYRAH_SSE_LOOP(i) {
        _mm_store_si128((__m128i*)&(dst[4*i]), _mm_castps_si128(data[i]));
      }
    }

    __m128 data[N/4];
  };

  template<int N>
  SYRAH_FORCEINLINE bool All(const FixedVectorMask<N, true>& v) {
    SYRAH_SSE_LOOP(i) {
#ifdef __SSE_4_1__
      if (!_mm_test_all_ones(_mm_castps_si128(v.data[i]))) return false;
#else
      if (_mm_movemask_ps(v.data[i]) != 0xf) return false;
#endif
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE bool Any(const FixedVectorMask<N, true>& v) {
    SYRAH_SSE_LOOP(i) {
#ifdef __SSE_4_1__
      if (!_mm_test_all_zeros(_mm_castps_si128(v.data[i]),
                              _mm_castps_si128(v.data[i]))) return true;
#else
      if (_mm_movemask_ps(v.data[i]) != 0) return true;
#endif
    }
    return false;
  }

  template<int N>
  SYRAH_FORCEINLINE bool None(const FixedVectorMask<N, true>& v) {
    SYRAH_SSE_LOOP(i) {
#ifdef __SSE_4_1__
      if (!_mm_test_all_zeros(_mm_castps_si128(v.data[i]),
                              _mm_castps_si128(v.data[i]))) return false;
#else
      if (_mm_movemask_ps(v.data[i]) != 0) return false;
#endif
    }
    return true;
  }

  // v1 & (!v2)
  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N> andNot(const FixedVectorMask<N, true>& v1, const FixedVectorMask<N, true>& v2) {
    FixedVectorMask<N> result;
    SYRAH_SSE_LOOP(i) {
      // NOTE(boulos): SSE applies the ! to the first argument.
      result.data[i] = _mm_andnot_ps(v2.data[i], v1.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> reverse(const FixedVectorMask<N, true>& a) {
    FixedVectorMask<N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_shuffle_ps(a.data[N/4 - 1 - i], a.data[N/4 - 1 - i], _MM_SHUFFLE(0, 1, 2, 3));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE int NumActive(const FixedVectorMask<N, true>& v) {
    int total_active = 0;
#ifdef __SSE_4_1__
    SYRAH_SSE_LOOP(i) {
      total_active += _mm_popcnt_u32(_mm_movemask_ps(v.data[i]));
    }
#else
    // Bits: 0/1 0/1 0/1 0/1
    // 0000 = 0 on
    // 0001 | 0010 | 0100 | 1000 = 1 on
    //
    // 0011 | 0101 | 1001
    // 0110 | 1010
    // 1100 = 2 on
    //
    // 0111 | 1011 | 1110 | 1101 = 3 on
    // 1111 = 4 on
    SYRAH_ALIGN(16) const int inc_amount[16] = {
      0, 1, 1, 2,
      1, 2, 2, 3,
      1, 2, 2, 3,
      2, 3, 3, 4
    };

    SYRAH_SSE_LOOP(i) {
      total_active += inc_amount[_mm_movemask_ps(v.data[i])];
    }
#endif
    return total_active;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> PrefixSum(const FixedVectorMask<N, true>& v) {
    FixedVector<int, N, true> result;
    int sum = 0;
    for (int i = 0; i < N; i++) {
      sum += (v.get(i)) ? 1 : 0;
      result[i] = sum;
    }
    return result;
  }



#undef SYRAH_SSE_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_SSE_MASK_H_
