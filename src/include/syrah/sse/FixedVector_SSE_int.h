#ifndef _SYRAH_FIXED_VECTOR_SSE_INT_H_
#define _SYRAH_FIXED_VECTOR_SSE_INT_H_

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(16) FixedVector<int, N, true> {
    public:
#define SYRAH_SSE_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)

    SYRAH_FORCEINLINE FixedVector() {}

    SYRAH_FORCEINLINE FixedVector(const int value) {
      load(value);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values) {
      load(values);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values, bool aligned) {
      load_aligned(values);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values, const FixedVectorMask<N>& mask,
                                 const int default_value) {
      load(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const int* values, const FixedVectorMask<N>& mask,
                                 const int default_value, bool aligned) {
      load_aligned(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<double, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_set_epi32(static_cast<int>(v[4*i+3]),
                                static_cast<int>(v[4*i+2]),
                                static_cast<int>(v[4*i+1]),
                                static_cast<int>(v[4*i+0]));
      }
    }

    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<float, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        // NOTE(boulos): Truncation, not rounding mode (round-to-nearest)
        data[i] = _mm_cvttps_epi32(v.data[i]);
      }
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<int, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    static SYRAH_FORCEINLINE FixedVector<int, N, true> reinterpret(const FixedVector<float, N, true>& v);

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator=(const FixedVector<int, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    static SYRAH_FORCEINLINE FixedVector<int, N, true> Zero() {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_setzero_si128();
      }
      return result;
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
      __m128i splat = _mm_set1_epi32(value);
      SYRAH_SSE_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const int* values) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          data[i] = _mm_load_si128((__m128i*)&values[4*i]);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_SSE_LOOP(i) {
          data[i] = _mm_loadu_si128((__m128i*)&values[4*i]);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const int* values) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_load_si128((__m128i*)&values[4*i]);
      }
    }

    SYRAH_FORCEINLINE void load(const int* values,
                                const FixedVectorMask<N>& mask,
                                const int default_value) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 16);
      __m128i default_data = _mm_set1_epi32(default_value);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          __m128i loaded_data = _mm_load_si128((__m128i*)&(values[4*i]));
          data[i] = syrah_blendv_int(loaded_data, default_data, _mm_castps_si128(mask.data[i]));
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_SSE_LOOP(i) {
          __m128i loaded_data = _mm_loadu_si128((__m128i*)&(values[4*i]));
          data[i] = syrah_blendv_int(loaded_data, default_data, _mm_castps_si128(mask.data[i]));
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const int* values,
                                        const FixedVectorMask<N>& mask,
                                        const int default_value) {
      __m128i default_data = _mm_set1_epi32(default_value);
      SYRAH_SSE_LOOP(i) {
        __m128i loaded_data = _mm_load_si128((__m128i*)&(values[4*i]));
        data[i] = syrah_blendv_int(loaded_data, default_data, _mm_castps_si128(mask.data[i]));
      }
    }

    template <typename T>
    SYRAH_FORCEINLINE void gather(const T* base, const FixedVector<int, N, true>& offsets, const int scale) {
      int* int_data = reinterpret_cast<int*>(data);
      // TODO(boulos): Consider doing all the indexing ops with
      // vector? Seems pointless.
      for (int i = 0; i < N; i++) {
        const T* addr = reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
        int_data[i] = static_cast<int>(*addr);
      }
    }

    // QUESTION(boulos): Constant scale only?
    template <typename T>
    SYRAH_FORCEINLINE void gather(const T* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      int* int_data = reinterpret_cast<int*>(data);
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          const T* addr = reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
          int_data[i] = static_cast<int>(*addr);
        }
      }
    }

    SYRAH_FORCEINLINE void store(int* dst) const {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          _mm_store_si128((__m128i*)&(dst[4*i]), data[i]);
        }
      } else {
        SYRAH_SSE_LOOP(i) {
          _mm_storeu_si128((__m128i*)&(dst[4*i]), data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst) const {
      SYRAH_SSE_LOOP(i) {
        _mm_store_si128((__m128i*)&(dst[4*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store(int* dst, const FixedVectorMask<N>& mask) const {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          // TODO(boulos): Consider maskmove? (was slower for float
          // store)
          __m128i cur_val = _mm_load_si128((__m128i*)&(dst[4*i]));
          _mm_store_si128((__m128i*)&(dst[4*i]), syrah_blendv_int(data[i], cur_val, _mm_castps_si128(mask.data[i])));
        }
      } else {
        SYRAH_SSE_LOOP(i) {
          __m128i cur_val = _mm_loadu_si128((__m128i*)&(dst[4*i]));
          _mm_storeu_si128((__m128i*)&(dst[4*i]), syrah_blendv_int(data[i], cur_val, _mm_castps_si128(mask.data[i])));
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_SSE_LOOP(i) {
        // Consider maskmov
        __m128i cur_val = _mm_load_si128((__m128i*)&(dst[4*i]));
        _mm_store_si128((__m128i*)&(dst[4*i]), syrah_blendv_int(data[i], cur_val, _mm_castps_si128(mask.data[i])));
      }
    }

    SYRAH_FORCEINLINE void scatter(int* base, const FixedVector<int, N, true>& offsets, const int scale) const {
      const int* int_data = reinterpret_cast<const int*>(data);
      // TODO(boulos): Consider doing all the indexing ops with
      // vector? Seems pointless.
      for (int i = 0; i < N; i++) {
        int* addr = reinterpret_cast<int*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
        *addr = int_data[i];
      }
    }

    // QUESTION(boulos): Constant scale only?
    SYRAH_FORCEINLINE void scatter(int* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) const {
      const int* int_data = reinterpret_cast<const int*>(data);
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          int* addr = reinterpret_cast<int*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
          *addr = int_data[i];
        }
      }
    }


    SYRAH_FORCEINLINE void merge(const FixedVector<int, N, true>& v,
                                 const FixedVectorMask<N, true>& mask) {
      SYRAH_SSE_LOOP(i) {
        data[i] = syrah_blendv_int(v.data[i], data[i], _mm_castps_si128(mask.data[i]));
      }
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator+(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_add_epi32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator+=(const FixedVector<int, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_add_epi32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator-() const {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_sub_epi32(_mm_setzero_si128(), data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator-(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_sub_epi32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator-=(const FixedVector<int, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_sub_epi32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator*(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = syrah_mm_mul_epi32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator*=(const FixedVector<int, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = syrah_mm_mul_epi32(data[i], v.data[i]);
      }
      return *this;
    }


    SYRAH_FORCEINLINE FixedVector<int, N, true> operator/(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      const int* int_data = reinterpret_cast<const int*>(data);
      for (int i = 0; i < N; i++) {
        result[i] = int_data[i] / v[i];
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator/=(const FixedVector<int, N, true>& v) {
      int* int_data = reinterpret_cast<int*>(data);
      for (int i = 0; i < N; i++) {
        int_data[i] /= v[i];
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator%(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      const int* int_data = reinterpret_cast<const int*>(data);
      for (int i = 0; i < N; i++) {
        result[i] = int_data[i] % v[i];
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator%=(const FixedVector<int, N, true>& v) {
      int* int_data = reinterpret_cast<int*>(data);
      for (int i = 0; i < N; i++) {
        int_data[i] %= v[i];
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator&(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_and_si128(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator&(const int& v) const {
      FixedVector<int, N, true> result;
      const __m128i splat = _mm_set1_epi32(v);
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_and_si128(data[i], splat);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator&=(const FixedVector<int, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_and_si128(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator&=(const int& v) {
      const __m128i splat = _mm_set1_epi32(v);
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_and_si128(data[i], splat);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator<<(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      const int* int_data = reinterpret_cast<const int*>(data);
      for (int i = 0; i < N; i++) {
        result[i] = int_data[i] << v[i];
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator<<=(const FixedVector<int, N, true>& v) {
      int* int_data = reinterpret_cast<int*>(data);
      for (int i = 0; i < N; i++) {
        int_data[i] <<= v[i];
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator<<(int v) const {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_slli_epi32(data[i], v);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator<<=(int v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_slli_epi32(data[i], v);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator>>(int v) const {
      FixedVector<int, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_srai_epi32(data[i], v);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator>>(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      const int* int_data = reinterpret_cast<const int*>(data);
      for (int i = 0; i < N; i++) {
        result[i] = int_data[i] >> v[i];
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator>>=(int v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_srai_epi32(data[i], v);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator>>=(const FixedVector<int, N, true>& v) {
      int* int_data = reinterpret_cast<int*>(data);
      for (int i = 0; i < N; i++) {
        int_data[i] >>= v[i];
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator|(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_or_si128(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator|=(const FixedVector<int, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_or_si128(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator^(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_xor_si128(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator^=(const FixedVector<int, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_xor_si128(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_castsi128_ps(_mm_cmplt_epi32(data[i], v.data[i]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_castsi128_ps(_mm_or_si128(_mm_cmplt_epi32(data[i], v.data[i]),
                                                       _mm_cmpeq_epi32(data[i], v.data[i])));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_castsi128_ps(_mm_cmpeq_epi32(data[i], v.data[i]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      // Hard to say if this is faster than a cast from cmpeq(setzero, setzero)...
      const __m128i all_ones = _mm_set1_epi32(-1);
      SYRAH_SSE_LOOP(i) {
        // NOT= & TRUE => !=
        result.data[i] = _mm_castsi128_ps(_mm_andnot_si128(_mm_cmpeq_epi32(data[i], v.data[i]), all_ones));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_castsi128_ps(_mm_or_si128(_mm_cmpgt_epi32(data[i], v.data[i]),
                                                       _mm_cmpeq_epi32(data[i], v.data[i])));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_castsi128_ps(_mm_cmpgt_epi32(data[i], v.data[i]));
      }
      return result;
    }

    SYRAH_FORCEINLINE int MaxElement() const {
      __m128i tmpMax = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMax = syrah_mm_max_epi32(data[i], tmpMax);
      }
      return syrah_mm_hmax_epi32(tmpMax);
    }

    SYRAH_FORCEINLINE int MinElement() const {
      __m128i tmpMin = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMin = syrah_mm_min_epi32(data[i], tmpMin);
      }
      return syrah_mm_hmin_epi32(tmpMin);
    }

    SYRAH_FORCEINLINE int foldMax() const { return MaxElement(); }
    SYRAH_FORCEINLINE int foldMin() const { return MinElement(); }

    SYRAH_FORCEINLINE int foldSum() const {
      __m128i tmp_sum = _mm_setzero_si128();
      SYRAH_SSE_LOOP(i) {
        tmp_sum = _mm_add_epi32(tmp_sum, data[i]);
      }
      // Have [a, b, c, d] want a + b + c + d
      __m128i ab_cd_ab_cd = _mm_hadd_epi32(tmp_sum, _mm_setzero_si128()); // [a+b, c+d, 0, 0]
      __m128i abcd = _mm_hadd_epi32(ab_cd_ab_cd, _mm_setzero_si128()); // [a+b+c+d x2, 0, 0]
      return _mm_cvtsi128_si32(abcd);
    }

    SYRAH_FORCEINLINE int foldProd() const {
      __m128i tmp_prod = _mm_set1_epi32(1);
      SYRAH_SSE_LOOP(i) {
        tmp_prod = syrah_mm_mul_epi32(tmp_prod, data[i]);
      }
      // Have [a, b, c, d] want a * b * c * d

      // temp1 = [a, a, b, b] * [c, c, d, d] = [a * c, a * c, b * d, b * d]
      __m128i temp1 = syrah_mm_mul_epi32(_mm_unpacklo_epi32(tmp_prod, tmp_prod), _mm_unpackhi_epi32(tmp_prod, tmp_prod));
      // Get b*d out
      __m128i bd = _mm_shuffle_epi32(temp1, _MM_SHUFFLE(2, 2, 2, 2));
      __m128i abcd = syrah_mm_mul_epi32(temp1, bd);
      return _mm_cvtsi128_si32(abcd);
    }

    __m128i data[N/4];
  };

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator+(const int& s, const FixedVector<int, N, true>& v) {
    FixedVector<int, N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_add_epi32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator+(const FixedVector<int, N, true>& v, const int& s) {
    FixedVector<int, N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_add_epi32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator-(const int& s, const FixedVector<int, N, true>& v) {
    FixedVector<int, N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_sub_epi32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator-(const FixedVector<int, N, true>& v, const int& s) {
    FixedVector<int, N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_sub_epi32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator*(const int& s, const FixedVector<int, N, true>& v) {
    FixedVector<int, N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = syrah_mm_mul_epi32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator*(const FixedVector<int, N, true>& v, const int& s) {
    FixedVector<int, N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = syrah_mm_mul_epi32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator/(const int& s, const FixedVector<int, N, true>& v) {
    FixedVector<int, N> result;
    for (int i = 0; i < N; i++) {
      result[i] = s / v[i];
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator/(const FixedVector<int, N, true>& v, const int& s) {
    FixedVector<int, N> result;
    for (int i = 0; i < N; i++) {
      result[i] = v[i] / s;
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_cmplt_epi32(splat, v.data[i]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_cmplt_epi32(v.data[i], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_or_si128(_mm_cmplt_epi32(splat, v.data[i]),
                                                     _mm_cmpeq_epi32(splat, v.data[i])));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_or_si128(_mm_cmplt_epi32(v.data[i], splat),
                                                     _mm_cmpeq_epi32(v.data[i], splat)));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_cmpeq_epi32(v.data[i], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_cmpeq_epi32(splat, v.data[i]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_or_si128(_mm_cmpeq_epi32(splat, v.data[i]),
                                                     _mm_cmplt_epi32(v.data[i], splat)));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_or_si128(_mm_cmpeq_epi32(v.data[i], splat),
                                                     _mm_cmplt_epi32(splat, v.data[i])));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_cmplt_epi32(v.data[i], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_castsi128_ps(_mm_cmplt_epi32(splat, v.data[i]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    const __m128i all_ones = _mm_set1_epi32(-1);
    SYRAH_SSE_LOOP(i) {
      // NOT= & TRUE => !=
      result.data[i] = _mm_castsi128_ps(_mm_andnot_si128(_mm_cmpeq_epi32(splat, v.data[i]), all_ones));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const __m128i splat = _mm_set1_epi32(s);
    const __m128i all_ones = _mm_set1_epi32(-1);
    SYRAH_SSE_LOOP(i) {
      // NOT= & TRUE => !=
      result.data[i] = _mm_castsi128_ps(_mm_andnot_si128(_mm_cmpeq_epi32(v.data[i], splat), all_ones));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> max(const FixedVector<int, N, true>& v1, const FixedVector<int, N, true>& v2) {
    FixedVector<int, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = syrah_mm_max_epi32(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> min(const FixedVector<int, N, true>& v1, const FixedVector<int, N, true>& v2) {
    FixedVector<int, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = syrah_mm_min_epi32(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> abs(const FixedVector<int, N, true>& v) {
    FixedVector<int, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_abs_epi32(v.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> madd(const FixedVector<int, N, true>& a,
                                        const FixedVector<int, N, true>& b,
                                        const FixedVector<int, N, true>& c) {
    return (a * b + c);
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
    SYRAH_SSE_LOOP(i) {
      result.data[i] = syrah_blendv_int(a.data[i], b.data[i], _mm_castps_si128(mask.data[i]));
    }
    return result;
  }

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

#undef SYRAH_SSE_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_SSE_INT_H_
