#ifndef _SYRAH_FIXED_VECTOR_AVX_MASK_H_
#define _SYRAH_FIXED_VECTOR_AVX_MASK_H_

namespace syrah {
  // NOTE(boulos): Could instead have this represented as one bit per
  // element and have operator< take _mm256_movemask_ps and cast to exact
  // bits each time...
  template<int N>
  class SYRAH_ALIGN(32) FixedVectorMask<N, true> {
  public:
#define SYRAH_AVX_LOOP(index) SYRAH_UNROLL(N/8) \
for (int index = 0; index < N/8; index++)

    SYRAH_FORCEINLINE FixedVectorMask(bool value) {
      if (true) {
        __m256 all_on = _mm256_cmpeq_ps(_mm256_setzero_ps(), _mm256_setzero_ps());
        SYRAH_AVX_LOOP(i) {
          data[i] = all_on;
        }
      } else {
        SYRAH_AVX_LOOP(i) {
          data[i] = _mm256_setzero_ps();
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
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_load_ps((float*)&values[8*i]);
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask(const FixedVectorMask<N>& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator=(const FixedVectorMask<N>& v) {
      SYRAH_AVX_LOOP(i) {
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
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_or_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator|=(const FixedVectorMask& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_or_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator&(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_and_ps(data[i],v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator&=(const FixedVectorMask& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_and_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator^(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_xor_ps(data[i],v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator^=(const FixedVectorMask& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_xor_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE static FixedVectorMask True() {
      // NOTE(boulos): Don't just call return FixedVectorMask(true);
      // to avoid branch.
      FixedVectorMask result;
      __m256 all_on = _mm256_cmpeq_ps(_mm256_setzero_ps(), _mm256_setzero_ps());
      SYRAH_AVX_LOOP(i) {
        result.data[i] = all_on;
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask False() {
      FixedVectorMask result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_setzero_ps();
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask FirstN(int n) {
      __m128i n_splat = _mm_set1_epi32(n);
      __m128i seq0 = _mm_set_epi32(3, 2, 1, 0);
      __m128i seq1 = _mm_set_epi32(7, 6, 5, 4);
      __m128i inc_val = _mm_set1_epi32(8);
      FixedVectorMask result;
      SYRAH_AVX_LOOP(i) {
        __m128i mask0 = _mm_cmplt_epi32(seq0, n_splat);
        __m128i mask1 = _mm_cmplt_epi32(seq1, n_splat);
        __m256i merged = _mm256_insertf128_si256(_mm256_castsi128_si256(mask0), mask1, 1);
        result.data[i] = _mm256_castsi256_ps(merged);
        seq0 = _mm_add_epi32(seq0, inc_val);
        seq1 = _mm_add_epi32(seq1, inc_val);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator!() const {
      FixedVectorMask result;
      __m256 all_on = _mm256_cmpeq_ps(_mm256_setzero_ps(), _mm256_setzero_ps());
      SYRAH_AVX_LOOP(i) {
        // NOTE(boulos): AVX's andnot function is actually (NOT x) & y
        result.data[i] = _mm256_andnot_ps(data[i], all_on);
      }
      return result;
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst) const {
      SYRAH_AVX_LOOP(i) {
        _mm256_store_ps(&(dst[8*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst) const {
      SYRAH_AVX_LOOP(i) {
        _mm256_store_si256((__m256i*)&(dst[8*i]), _mm256_castps_si256(data[i]));
      }
    }

    __m256 data[N/8];
  };

  template<int N>
  SYRAH_FORCEINLINE bool All(const FixedVectorMask<N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      if (_mm256_movemask_ps(v.data[i]) != 0xff) return false;
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE bool Any(const FixedVectorMask<N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      if (_mm256_movemask_ps(v.data[i]) != 0) return true;
    }
    return false;
  }

  template<int N>
  SYRAH_FORCEINLINE bool None(const FixedVectorMask<N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      if (_mm256_movemask_ps(v.data[i]) != 0) return false;
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE int NumActive(const FixedVectorMask<N, true>& v) {
    int total_active = 0;
    SYRAH_AVX_LOOP(i) {
      total_active += _mm_popcnt_u32(_mm256_movemask_ps(v.data[i]));
    }
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

#undef SYRAH_AVX_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_AVX_MASK_H_
