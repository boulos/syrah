#ifndef _SYRAH_FIXED_VECTOR_AVX_DOUBLE_H_
#define _SYRAH_FIXED_VECTOR_AVX_DOUBLE_H_

namespace syrah {
  // Take the two double masks and collapse them to a single precision mask
  SYRAH_FORCEINLINE __m256 CollapseDoubles(__m256d a, __m256d b) {
    __m128 lo_a = _mm_castpd_ps(_mm256_extractf128_pd(a, 0)); // a_f[0,1] and a_f[2,3]
    __m128 hi_a = _mm_castpd_ps(_mm256_extractf128_pd(a, 1)); // a_f[4,5] and a_f[6,7]

    // even pairs
    __m128 mix_a = _mm_shuffle_ps(lo_a, hi_a, _MM_SHUFFLE(2, 0, 2, 0));

    // same for b
    __m128 lo_b = _mm_castpd_ps(_mm256_extractf128_pd(b, 0));
    __m128 hi_b = _mm_castpd_ps(_mm256_extractf128_pd(b, 1));
    __m128 mix_b = _mm_shuffle_ps(lo_b, hi_b, _MM_SHUFFLE(2, 0, 2, 0));

    // merge them back
    return _mm256_insertf128_ps(_mm256_castps128_ps256(mix_a), mix_b, 1);
  }

  // Undo the above
  __m256d ExpandToDouble(__m256 a, bool first) {
    // either first 4 or last 4

    // LAME(boulos): This generates ICE for ICC 12.0 on bringup

    //__m128 subset = (first) ? _mm256_extractf128_ps(a, 0) : _mm256_extractf128_ps(a, 1);

    __m256 subset_256 = (first) ? a : _mm256_permute2f128_ps(a, a, 0x1);
    __m128 subset = _mm256_castps256_ps128(subset_256);

    __m128 lo_part = _mm_shuffle_ps(subset, subset, _MM_SHUFFLE(1, 1, 0, 0));
    __m128 hi_part = _mm_shuffle_ps(subset, subset, _MM_SHUFFLE(3, 3, 2, 2));
    return _mm256_castps_pd(_mm256_insertf128_ps(_mm256_castps128_ps256(lo_part), hi_part, 1));
  }

#define SYRAH_DOUBLE_MASK (((i & 1)) ? ExpandToDouble(mask.data[i/2], true) : ExpandToDouble(mask.data[i/2], false))


  template<int N>
  class SYRAH_ALIGN(32) FixedVector<double, N, true> {
    public:
#define SYRAH_AVX_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)

#define SYRAH_MASK_LOOP(index) SYRAH_UNROLL(N/8) \
for (int index = 0; index < N/8; index++)


    SYRAH_FORCEINLINE FixedVector() {}

    SYRAH_FORCEINLINE FixedVector(const double value) {
      load(value);
    }

    SYRAH_FORCEINLINE FixedVector(const double* values) {
      load(values);
    }

    SYRAH_FORCEINLINE FixedVector(const double* values, bool aligned) {
      load_aligned(values);
    }

    SYRAH_FORCEINLINE FixedVector(const double* values, const FixedVectorMask<N>& mask,
                                 const double default_value) {
      load(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const double* values, const FixedVectorMask<N>& mask,
                                 const double default_value, bool aligned) {
      load_aligned(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const double* base, const FixedVector<int, N, true>& offsets, const int scale) {
      gather(base, offsets, scale);
    }

    SYRAH_FORCEINLINE FixedVector(const double* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      gather(base, offsets, scale, mask);
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<float, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_set_pd(static_cast<double>(v[4*i+3]),
                                static_cast<double>(v[4*i+2]),
                                static_cast<double>(v[4*i+1]),
                                static_cast<double>(v[4*i+0]));
      }
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<int, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_set_pd(static_cast<double>(v[4*i+3]),
                                static_cast<double>(v[4*i+2]),
                                static_cast<double>(v[4*i+1]),
                                static_cast<double>(v[4*i+0]));
      }
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<double, N, true>& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator=(const FixedVector<double, N, true>& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    static SYRAH_FORCEINLINE FixedVector<double, N, true> Zero() {
      FixedVector<double, N, true> result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_setzero_pd();
      }
      return result;
    }

    SYRAH_FORCEINLINE const double& operator[](int i) const {
      return ((double*)data)[i];
    }

    SYRAH_FORCEINLINE double& operator[](int i) {
      return ((double*)data)[i];
    }

    SYRAH_FORCEINLINE void load(const double value) {
      __m256d splat = _mm256_set1_pd(value);
      SYRAH_AVX_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const double* values) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 32);
      if (sse_aligned) {
        SYRAH_AVX_LOOP(i) {
          data[i] = _mm256_load_pd(&values[4*i]);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_AVX_LOOP(i) {
          data[i] = _mm256_loadu_pd(&values[4*i]);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const double* values) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_load_pd(&values[4*i]);
      }
    }

    SYRAH_FORCEINLINE void load(const double* values, const FixedVectorMask<N>& mask, const double default_value) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 32);

      __m256d default_data = _mm256_set1_pd(default_value);
      if (sse_aligned) {
        SYRAH_AVX_LOOP(i) {
          __m256d mask_double = SYRAH_DOUBLE_MASK;
          __m256d loaded_data = _mm256_load_pd(&(values[4*i]));
          data[i] = _mm256_blendv_pd(default_data, loaded_data, mask_double);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_AVX_LOOP(i) {
          __m256d mask_double = SYRAH_EXPAND_TO_DOUBLE(mask.data[i/2], i & 1);
          __m256d loaded_data = _mm256_loadu_pd(&(values[4*i]));
          data[i] = _mm256_blendv_pd(default_data, loaded_data, mask_double);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const double* values, const FixedVectorMask<N>& mask, const double default_value) {
      __m256d default_data = _mm256_set1_pd(default_value);
      SYRAH_AVX_LOOP(i) {
        __m256d mask_double = SYRAH_DOUBLE_MASK;
        __m256d loaded_data = _mm256_load_pd(&(values[4*i]));
        data[i] = _mm256_blendv_pd(default_data, loaded_data, mask_double);
      }
    }

    SYRAH_FORCEINLINE void gather(const double* base, const FixedVector<int, N, true>& offsets, const int scale) {
      double* double_data = reinterpret_cast<double*>(data);
      for (int i = 0; i < N; i++) {
        const double* addr = reinterpret_cast<const double*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
        double_data[i] = *addr;
      }
    }

    SYRAH_FORCEINLINE void gather(const double* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      double* double_data = reinterpret_cast<double*>(data);
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          const double* addr = reinterpret_cast<const double*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
          double_data[i] = *addr;
        }
      }
    }

    SYRAH_FORCEINLINE void store(double* dst) const {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 32);
      if (sse_aligned) {
        SYRAH_AVX_LOOP(i) {
          _mm256_store_pd(&(dst[4*i]), data[i]);
        }
      } else {
        SYRAH_AVX_LOOP(i) {
          _mm256_storeu_pd(&(dst[4*i]), data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(double* dst) const {
      SYRAH_AVX_LOOP(i) {
        _mm256_store_pd(&(dst[4*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store(double* dst, const FixedVectorMask<N>& mask) const {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 32);
      if (sse_aligned) {
        SYRAH_AVX_LOOP(i) {
          __m256d mask_double = SYRAH_DOUBLE_MASK;
          __m256d cur_val = _mm256_load_pd(&(dst[4*i]));
          _mm256_store_pd(&(dst[4*i]), _mm256_blendv_pd(cur_val, data[i], mask_double));
        }
      } else {
        SYRAH_AVX_LOOP(i) {
          __m256d mask_double = SYRAH_DOUBLE_MASK;
          __m256d cur_val = _mm256_loadu_pd(&(dst[4*i]));
          _mm256_storeu_pd(&(dst[4*i]), _mm256_blendv_pd(cur_val, data[i], mask_double));
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(double* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_AVX_LOOP(i) {
        __m256d mask_double = SYRAH_DOUBLE_MASK;
        __m256d cur_val = _mm256_load_pd(&(dst[4*i]));
        _mm256_store_pd(&(dst[4*i]), _mm256_blendv_pd(cur_val, data[i], mask_double));
      }
    }


    SYRAH_FORCEINLINE void scatter(double* base, const FixedVector<int, N, true>& offsets, const int scale) const {
      const double* double_data = reinterpret_cast<const double*>(data);
      // TODO(boulos): Consider doing all the indexing ops with
      // vector? Seems pointless.
      for (int i = 0; i < N; i++) {
        double* addr = reinterpret_cast<double*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
        *addr = double_data[i];
      }
    }

    // QUESTION(boulos): Constant scale only?
    SYRAH_FORCEINLINE void scatter(double* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) const {
      const double* float_data = reinterpret_cast<const double*>(data);
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          double* addr = reinterpret_cast<double*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
          *addr = float_data[i];
        }
      }
    }

    SYRAH_FORCEINLINE void merge(const FixedVector<double, N, true>& v,
                                 const FixedVectorMask<N, true>& mask) {
      SYRAH_AVX_LOOP(i) {
        __m256d mask_double = SYRAH_DOUBLE_MASK;
        data[i] = _mm256_blendv_pd(data[i], v.data[i], mask_double);
      }
    }

    SYRAH_FORCEINLINE FixedVector<double, N> operator+(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_add_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N>& operator+=(const FixedVector<double, N, true>& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_add_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator-() const {
      FixedVector<double, N, true> result;
      const __m256d sign_mask = _mm256_set1_pd(-0.0);
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_xor_pd(data[i], sign_mask);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator-(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_sub_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator-=(const FixedVector<double, N, true>& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_sub_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator*(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_mul_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator*=(const FixedVector<double, N, true>& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_mul_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator/(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_AVX_LOOP(i) {
        result.data[i] = _mm256_div_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator/=(const FixedVector<double, N, true>& v) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_div_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm256_cmplt_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm256_cmplt_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm256_cmple_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm256_cmple_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm256_cmpeq_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm256_cmpeq_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm256_cmpge_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm256_cmpge_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm256_cmpgt_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm256_cmpgt_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE double MaxElement() const {
      __m256d tmpMax = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMax = _mm256_max_pd(data[i], tmpMax);
      }

      // Now we've got to reduce the 4-wide version into a single double.
      return syrah_mm256_hmax_pd(tmpMax);
    }

    SYRAH_FORCEINLINE double MinElement() const {
      __m256d tmpMin = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMin = _mm256_min_pd(data[i], tmpMin);
      }

      // Now we've got to reduce the 4-wide version into a single double.
      return syrah_mm256_hmin_pd(tmpMin);
    }

    SYRAH_FORCEINLINE double foldMax() const { return MaxElement(); }
    SYRAH_FORCEINLINE double foldMin() const { return MinElement(); }

    SYRAH_FORCEINLINE double foldSum() const {
      __m256d tmp_sum = _mm256_setzero_pd();
      SYRAH_AVX_LOOP(i) {
        tmp_sum = _mm256_add_pd(tmp_sum, data[i]);
      }
      // [a, b, c, d]

      // ab, 0, cd, 0
      __m256d ab_0_cd_0 = _mm256_hadd_pd(tmp_sum, _mm256_setzero_pd());
      // move cd into the 0-slot and add.
      __m256d final_sum = _mm256_add_pd(ab_0_cd_0, _mm256_permute2f128_pd(ab_0_cd_0, ab_0_cd_0, 0x1));
      return _mm_cvtsd_f64(_mm256_castpd256_pd128(final_sum));
    }

    SYRAH_FORCEINLINE double foldProd() const {
      __m256d tmp_prod = _mm256_set1_pd(1.0);
      SYRAH_AVX_LOOP(i) {
        tmp_prod = _mm256_mul_pd(tmp_prod, data[i]);
      }
      // Have [a, b, c, d] want a * b * c * d
      // temp1 = [a, b, c, d] * [b, b, d, d] = [ab, b^2, cd, d^2]
      __m256d temp1 = _mm256_mul_pd(tmp_prod, _mm256_unpackhi_pd(tmp_prod, tmp_prod));
      __m256d cd = _mm256_permute2f128_pd(temp1, temp1, 0x1);
      __m256d final_prod = _mm256_mul_pd(temp1, cd);
      return _mm_cvtsd_f64(_mm256_castpd256_pd128(final_prod));
    }

    __m256d data[N/4];
  };

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator+(const double& s, const FixedVector<double, N, true>& v) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_add_pd(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator+(const FixedVector<double, N, true>& v, const double& s) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_add_pd(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator-(const double& s, const FixedVector<double, N, true>& v) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_sub_pd(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator-(const FixedVector<double, N, true>& v, const double& s) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_sub_pd(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator*(const double& s, const FixedVector<double, N, true>& v) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_mul_pd(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator*(const FixedVector<double, N, true>& v, const double& s) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_mul_pd(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator/(const double& s, const FixedVector<double, N, true>& v) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_div_pd(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> operator/(const FixedVector<double, N, true>& v, const double& s) {
    FixedVector<double, N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_div_pd(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const double& s, const FixedVector<double, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_MASK_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmplt_pd(splat, v.data[2*i+0]),
                                       _mm256_cmplt_pd(splat, v.data[2*i+1]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const FixedVector<double, N, true>& v, const double& s) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmplt_pd(v.data[2*i+0], splat),
                                       _mm256_cmplt_pd(v.data[2*i+1], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const double& s, const FixedVector<double, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmple_pd(splat, v.data[2*i+0]),
                                       _mm256_cmple_pd(splat, v.data[2*i+1]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const FixedVector<double, N, true>& v, const double& s) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmple_pd(v.data[2*i+0], splat),
                                       _mm256_cmple_pd(v.data[2*i+1], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const FixedVector<double, N, true>& v, const double& s) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmpeq_pd(v.data[2*i+0], splat),
                                       _mm256_cmpeq_pd(v.data[2*i+1], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const double& s, const FixedVector<double, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmpge_pd(splat, v.data[2*i+0]),
                                       _mm256_cmpge_pd(splat, v.data[2*i+1]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const FixedVector<double, N, true>& v, const double& s) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmpge_pd(v.data[2*i+0], splat),
                                       _mm256_cmpge_pd(v.data[2*i+1], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const double& s, const FixedVector<double, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmpgt_pd(splat, v.data[2*i+0]),
                                       _mm256_cmpgt_pd(splat, v.data[2*i+1]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const FixedVector<double, N, true>& v, const double& s) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmpgt_pd(v.data[2*i+0], splat),
                                       _mm256_cmpgt_pd(v.data[2*i+1], splat));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const double& s, const FixedVector<double, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmpneq_pd(splat, v.data[2*i+0]),
                                       _mm256_cmpneq_pd(splat, v.data[2*i+1]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<double, N, true>& v, const double& s) {
    FixedVectorMask<N> result;
    const __m256d splat = _mm256_set1_pd(s);
    SYRAH_MASK_LOOP(i) {
      result.data[i] = CollapseDoubles(_mm256_cmpneq_pd(v.data[2*i+0], splat),
                                       _mm256_cmpneq_pd(v.data[2*i+1], splat));
    }
    return result;
  }



  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> max(const FixedVector<double, N, true>& v1, const FixedVector<double, N, true>& v2) {
    FixedVector<double, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_max_pd(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> min(const FixedVector<double, N, true>& v1, const FixedVector<double, N, true>& v2) {
    FixedVector<double, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_min_pd(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> sqrt(const FixedVector<double, N, true>& v1) {
    FixedVector<double, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_sqrt_pd(v1.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> madd(const FixedVector<double, N, true>& a,
                                                      const FixedVector<double, N, true>& b,
                                                      const FixedVector<double, N, true>& c) {
    FixedVector<double, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_add_pd(_mm256_mul_pd(a.data[i], b.data[i]), c.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> trunc(const FixedVector<double, N, true>& a) {
    FixedVector<double, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_round_pd(a.data[i], 3 /* round towards 0 = truncate */);
    }
    return result;
  }

  // result = round_to_float(a) (using current rounding mode)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> rint(const FixedVector<double, N, true>& a) {
    FixedVector<double, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_round_pd(a.data[i], 4 /* round using rounding mode */);
    }
    return result;
  }

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> select(const FixedVector<double, N, true>& a,
                                                        const FixedVector<double, N, true>& b,
                                                        const FixedVectorMask<N, true>& mask) {
    FixedVector<double, N, true> result;
    SYRAH_AVX_LOOP(i) {
      __m256d mask_double = SYRAH_DOUBLE_MASK;
      result.data[i] = _mm256_blendv_pd(b.data[i], a.data[i], mask_double);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> reverse(const FixedVector<double, N, true>& a) {
    FixedVector<double, N, true> result;
    // To reverse a N-wide vector, we first have to reverse the data
    // array and then the bits themselves.
    SYRAH_AVX_LOOP(i) {
      __m256d other = a.data[N/4 - i - 1];
      // Can't shuffle the whole thing with AVX, need to get the first half and the second half
      __m128d lo_part = _mm256_extractf128_pd(other, 0);
      __m128d hi_part = _mm256_extractf128_pd(other, 1);

      // Reverse the parts
      lo_part = _mm_shuffle_pd(lo_part, lo_part, _MM_SHUFFLE(0, 0, 1, 1));
      hi_part = _mm_shuffle_pd(hi_part, hi_part, _MM_SHUFFLE(0, 0, 1, 1));

      // Put it back together but in reverse part order (reversed_hi, reversed_lo)
      __m256d reversed = _mm256_insertf128_pd(_mm256_castpd128_pd256(hi_part), lo_part, 1);
      result.data[i] = reversed;
    }
    return result;
  }


#undef SYRAH_MASK_LOOP
#undef SYRAH_AVX_LOOP
#undef SYRAH_EXPAND_TO_DOUBLE
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_AVX_DOUBLE_H_
