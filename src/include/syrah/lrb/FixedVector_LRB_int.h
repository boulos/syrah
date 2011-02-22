#ifndef _SYRAH_FIXED_VECTOR_LRB_INT_H_
#define _SYRAH_FIXED_VECTOR_LRB_INT_H_

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(64) FixedVector<int, N, true> {
    public:
#define SYRAH_LRB_LOOP(index) SYRAH_UNROLL(N/16) \
for (int index = 0; index < N/16; index++)

    SYRAH_FORCEINLINE FixedVector() {}

    SYRAH_FORCEINLINE FixedVector(const int value) {
      load(value);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values) {
      load(values);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values, bool /*aligned*/) {
      load_aligned(values);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values, const FixedVectorMask<N>& mask,
                                 const int default_value) {
      load(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values, const FixedVectorMask<N>& mask,
                                 const int default_value, bool /*aligned*/) {
      load_aligned(values, mask, default_value);
    }

#if 0
    SYRAH_FORCEINLINE FixedVector(const FixedVector<double, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm_set_epi32(static_cast<int>(v[4*i+3]),
                                static_cast<int>(v[4*i+2]),
                                static_cast<int>(v[4*i+1]),
                                static_cast<int>(v[4*i+0]));
      }
    }
#endif

    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<float, N, true>& v);
    static SYRAH_FORCEINLINE FixedVector<int, N, true> reinterpret(const FixedVector<float, N, true>& v);

    static SYRAH_FORCEINLINE FixedVector<int, N, true> Zero() {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_setzero_pi();
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<int, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_LRB_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector(const int* base, const FixedVector<int, N, true>& offsets, const int scale) {
      gather(base, offsets, scale);
    }

    SYRAH_FORCEINLINE FixedVector(const int* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      gather(base, offsets, scale, mask);
    }

    SYRAH_FORCEINLINE FixedVector(const char* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      gather(base, offsets, scale, mask);
    }

    SYRAH_FORCEINLINE const int& operator[](int i) const {
      return ((int*)data)[i];
    }

    SYRAH_FORCEINLINE int& operator[](int i) {
      return ((int*)data)[i];
    }

    SYRAH_FORCEINLINE void load(const int value) {
      __m512i splat = _mm512_set_1to16_pi(value);
      SYRAH_LRB_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const int* values) {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(values, 64);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_castps_si512(_mm512_loadd(const_cast<int*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE));
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_loadu_epi32(const_cast<int*>(&(values[16*i])));
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const int* values) {
       SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_castps_si512(_mm512_loadd(const_cast<int*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE));
       }
    }

    SYRAH_FORCEINLINE void load(const int* values, const FixedVectorMask<N>& mask,
                               const int default_value) {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(values, 64);
      __m512i default_data = _mm512_set_1to16_pi(default_value);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_castps_si512(_mm512_mask_loadd(default_data, mask.data[i], const_cast<int*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE));
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_mask_loadu_epi32(const_cast<int*>(&(values[16*i])), mask.data[i], default_data);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const int* values, const FixedVectorMask<N>& mask,
                                       const int default_value) {
      __m512i default_data = _mm512_set_1to16_pi(default_value);
      SYRAH_LRB_LOOP(i) {
         data[i] = _mm512_castps_si512(_mm512_mask_loadd(default_data, mask.data[i], const_cast<int*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE));
      }
    }

    // TODO(boulos): Actually, I don't think this works correctly for
    // T!=int. We need to set the UPC bits.
    template <typename T>
    SYRAH_FORCEINLINE void gather(const T* base, const FixedVector<int, N, true>& offsets, const int scale) {
#define SYRAH_GATHER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { data[i] = _mm512_castps_si512(_mm512_gatherd(offsets.data[i], const_cast<T*>(base), _MM_FULLUPC_NONE, SCALE, _MM_HINT_NONE)); }
      switch (scale) {
      case 1: SYRAH_GATHER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_GATHER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_GATHER_LOOP(_MM_SCALE_4); break;
      default: SYRAH_GATHER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_GATHER_LOOP
    }

    // QUESTION(boulos): Constant scale only?
    template <typename T>
    SYRAH_FORCEINLINE void gather(const T* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
#define SYRAH_GATHER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { data[i] = _mm512_castps_si512(_mm512_mask_gatherd(_mm512_castsi512_ps(data[i]), mask.data[i], offsets.data[i], const_cast<T*>(base), _MM_FULLUPC_NONE, SCALE, _MM_HINT_NONE)); }
      switch (scale) {
      case 1: SYRAH_GATHER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_GATHER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_GATHER_LOOP(_MM_SCALE_4); break;
      default: SYRAH_GATHER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_GATHER_LOOP
    }

    SYRAH_FORCEINLINE void store(int* dst) const {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(dst, 64);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          _mm512_stored(&(dst[16*i]), data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
        }
      } else {
        SYRAH_LRB_LOOP(i) {
          _mm512_storeu_epi32(&(dst[16*i]), data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst) const {
       SYRAH_LRB_LOOP(i) {
          _mm512_stored(&(dst[16*i]), data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
       }
    }

    SYRAH_FORCEINLINE void store_aligned_stream(int* dst) const {
       SYRAH_LRB_LOOP(i) {
          _mm512_stored(&(dst[16*i]), data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NT);
       }
    }

    SYRAH_FORCEINLINE void store(int* dst, const FixedVectorMask<N>& mask) const {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(dst, 64);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          _mm512_mask_stored(&(dst[16*i]), mask.data[i], data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
        }
      } else {
        SYRAH_LRB_LOOP(i) {
          _mm512_mask_storeu_epi32(&(dst[16*i]), mask.data[i], data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst, const FixedVectorMask<N>& mask) const {
       SYRAH_LRB_LOOP(i) {
          _mm512_mask_stored(&(dst[16*i]), mask.data[i], data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
       }
    }

    SYRAH_FORCEINLINE void store_aligned_stream(int* dst, const FixedVectorMask<N>& mask) const {
       SYRAH_LRB_LOOP(i) {
          _mm512_mask_stored(&(dst[16*i]), mask.data[i], data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NT);
       }
    }


    SYRAH_FORCEINLINE void scatter(int* base, const FixedVector<int, N, true>& offsets, const int scale) const {
#define SYRAH_SCATTER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { _mm512_scatterd(base, offsets.data[i], _mm512_castsi512_ps(data[i]), _MM_DOWNC_NONE, SCALE, _MM_HINT_NONE); }
      switch (scale) {
      case 1: SYRAH_SCATTER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_SCATTER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_SCATTER_LOOP(_MM_SCALE_4); break;
      default : SYRAH_SCATTER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_SCATTER_LOOP
    }

    // QUESTION(boulos): Constant scale only?
    SYRAH_FORCEINLINE void scatter(int* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) const {
#define SYRAH_SCATTER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { _mm512_mask_scatterd(base, mask.data[i], offsets.data[i], _mm512_castsi512_ps(data[i]), _MM_DOWNC_NONE, SCALE, _MM_HINT_NONE); }
      switch (scale) {
      case 1: SYRAH_SCATTER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_SCATTER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_SCATTER_LOOP(_MM_SCALE_4); break;
      default : SYRAH_SCATTER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_SCATTER_LOOP
    }


    SYRAH_FORCEINLINE void merge(const FixedVector<int, N, true>& v,
                                const FixedVectorMask<N, true>& mask) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_castps_si512(_mm512_mask_movd(_mm512_castsi512_ps(data[i]),
                                                       mask.data[i],
                                                       _mm512_castsi512_ps(v.data[i])));
      }
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator+(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_add_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator+=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_add_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator-() const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_sub_pi(_mm512_setzero_pi(), data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator-(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_sub_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator-=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_sub_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator*(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_mull_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator*=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_mull_pi(data[i], v.data[i]);
      }
      return *this;
    }


    SYRAH_FORCEINLINE FixedVector<int, N, true> operator/(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_div_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator/=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_div_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator%(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_rem_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator%=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_rem_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator&(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_and_pi(data[i], v.data[i]);
      }
      return result;
    }

    // QUESTION(boulos): Will adding an explicit "constant int"
    // version allow me to improve this?
    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator&=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_and_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator<<(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_sll_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator<<=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_sll_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator<<(int v) const {
      FixedVector<int, N, true> result;
      __m512i shift_amt = _mm512_set_1to16_pi(v);
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_sll_pi(data[i], shift_amt);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator<<=(int v) {
      __m512i shift_amt = _mm512_set_1to16_pi(v);
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_sll_pi(data[i], shift_amt);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator>>(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_sra_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator>>=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_sra_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator>>(int v) const {
      FixedVector<int, N, true> result;
      __m512i shift_amt = _mm512_set_1to16_pi(v);
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_sra_pi(data[i], shift_amt);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator>>=(int v) {
      __m512i shift_amt = _mm512_set_1to16_pi(v);
      SYRAH_LRB_LOOP(i) {
         data[i] = _mm512_sra_pi(data[i], shift_amt);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator|=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_or_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator|(const FixedVector<int, N, true>& v) const {
       FixedVector<int, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_or_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator^(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_xor_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator^=(const FixedVector<int, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_xor_pi(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmplt_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmple_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmpeq_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmpneq_pi(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      // As in floats, a >= b -> b <= a
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmple_pi(v.data[i], data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      // a > b <-> b < a
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmplt_pi(v.data[i], data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE int MinElement() const {
      __m512i tmpMin = data[0];
      for (int i = 1; i < N/16; i++) {
        tmpMin = _mm512_min_pi(data[i], tmpMin);
      }
      return _mm512_reduce_min_epi32(tmpMin);
    }


    SYRAH_FORCEINLINE int MaxElement() const {
      __m512i tmpMax = data[0];
      for (int i = 1; i < N/16; i++) {
        tmpMax = _mm512_max_pi(data[i], tmpMax);
      }
      return _mm512_reduce_max_epi32(tmpMax);
    }

    SYRAH_FORCEINLINE int foldMin() const { return MinElement(); }
    SYRAH_FORCEINLINE int foldMax() const { return MaxElement(); }
    SYRAH_FORCEINLINE int foldSum() const {
      __m512i result = _mm512_setzero_pi();
      SYRAH_LRB_LOOP(i) {
        result = _mm512_add_pi(result, data[i]);
      }
      return _mm512_reduce_add_pi(result);
    }

    SYRAH_FORCEINLINE int foldProd() const {
      __m512i result = _mm512_set_1to16_pi(1);
      SYRAH_LRB_LOOP(i) {
        result = _mm512_mull_pi(result, data[i]);
      }
      return _mm512_reduce_mul_pi(result);
    }

    __m512i data[N/16];
  };

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> max(const FixedVector<int, N, true>& v1, const FixedVector<int, N, true>& v2) {
    FixedVector<int, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_max_pi(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> min(const FixedVector<int, N, true>& v1, const FixedVector<int, N, true>& v2) {
    FixedVector<int, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_min_pi(v1.data[i], v2.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> madd(const FixedVector<int, N, true>& a,
                                        const FixedVector<int, N, true>& b,
                                        const FixedVector<int, N, true>& c) {
    FixedVector<int, N, true> result;
    SYRAH_LRB_LOOP(i) {
      // XXX(boulos): Why isn't there a 213_pi?
      result.data[i] = _mm512_madd231_pi(c.data[i], a.data[i], b.data[i]);
    }
    return result;
  }

  // truncate(a) = a
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> trunc(const FixedVector<int, N, true>& a) {
    return a;
  }

  // rint(a) = a
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> rint(const FixedVector<int, N, true>& a) {
    return a;
  }

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> select(const FixedVector<int, N, true>& a,
                                                    const FixedVector<int, N, true>& b,
                                                    const FixedVectorMask<N>& mask) {
    FixedVector<int, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_castps_si512(_mm512_mask_movd(_mm512_castsi512_ps(b.data[i]),
                                                            mask.data[i],
                                                            _mm512_castsi512_ps(a.data[i])));
    }
    return result;
  }

#undef SYRAH_LRB_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_LRB_INT_H_
