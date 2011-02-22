#ifndef _SYRAH_FIXED_VECTOR_SSE_DOUBLE_H_
#define _SYRAH_FIXED_VECTOR_SSE_DOUBLE_H_

namespace syrah {
  // Take a pair of doubles and extract the 1st and 3rd word from each.
  SYRAH_FORCEINLINE __m128 CollapseDoubles(__m128d a, __m128d b) {
    return _mm_shuffle_ps(_mm_castpd_ps(a),
                          _mm_castpd_ps(b),
                          _MM_SHUFFLE(2, 0, 2, 0));
  }
  // Undo the above
  SYRAH_FORCEINLINE __m128d ExpandToDouble(__m128 a, int which) {
    return (which == 0) ?
      _mm_castps_pd(_mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 0, 0))) :
      _mm_castps_pd(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 2, 2)));
  }

  template<int N>
  class SYRAH_ALIGN(16) FixedVector<double, N, true> {
    public:
#define SYRAH_SSE_LOOP(index) SYRAH_UNROLL(N/2) \
for (int index = 0; index < N/2; index++)

#define SYRAH_MASK_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)


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
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_set_pd(static_cast<double>(v[2*i+1]),
                             static_cast<double>(v[2*i+0]));
      }
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<int, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_set_pd(static_cast<double>(v[2*i+1]),
                             static_cast<double>(v[2*i+0]));
      }
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<double, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator=(const FixedVector<double, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    static SYRAH_FORCEINLINE FixedVector<double, N, true> Zero() {
      FixedVector<double, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_setzero_pd();
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
      __m128d splat = _mm_set1_pd(value);
      SYRAH_SSE_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const double* values) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          data[i] = _mm_load_pd(&values[2*i]);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_SSE_LOOP(i) {
          data[i] = _mm_loadu_pd(&values[2*i]);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const double* values) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_load_pd(&values[2*i]);
      }
    }

    SYRAH_FORCEINLINE void load(const double* values, const FixedVectorMask<N>& mask, const double default_value) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 16);

      __m128d default_data = _mm_set1_pd(default_value);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
          __m128d loaded_data = _mm_load_pd(&(values[2*i]));
          data[i] = syrah_blendv_pd(loaded_data, default_data, mask_double);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_SSE_LOOP(i) {
          __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
          __m128d loaded_data = _mm_loadu_pd(&(values[2*i]));
          data[i] = syrah_blendv_pd(loaded_data, default_data, mask_double);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const double* values, const FixedVectorMask<N>& mask, const double default_value) {
      __m128d default_data = _mm_set1_pd(default_value);
      SYRAH_SSE_LOOP(i) {
        __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
        __m128d loaded_data = _mm_load_pd(&(values[2*i]));
        data[i] = syrah_blendv_pd(loaded_data, default_data, mask_double);
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
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          _mm_store_pd(&(dst[2*i]), data[i]);
        }
      } else {
        SYRAH_SSE_LOOP(i) {
          _mm_storeu_pd(&(dst[2*i]), data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(double* dst) const {
      SYRAH_SSE_LOOP(i) {
        _mm_store_pd(&(dst[2*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store(double* dst, const FixedVectorMask<N>& mask) const {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
          __m128d cur_val = _mm_load_pd(&(dst[2*i]));
          _mm_store_pd(&(dst[2*i]), syrah_blendv_pd(data[i], cur_val, mask_double));
        }
      } else {
        SYRAH_SSE_LOOP(i) {
          __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
          __m128d cur_val = _mm_loadu_pd(&(dst[2*i]));
          _mm_storeu_pd(&(dst[2*i]), syrah_blendv_pd(data[i], cur_val, mask_double));
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(double* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_SSE_LOOP(i) {
        __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
        __m128d cur_val = _mm_load_pd(&(dst[2*i]));
        _mm_store_pd(&(dst[2*i]), syrah_blendv_pd(data[i], cur_val, mask_double));
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
      SYRAH_SSE_LOOP(i) {
        __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
        data[i] = syrah_blendv_pd(v.data[i], data[i], mask_double);
      }
    }

    SYRAH_FORCEINLINE FixedVector<double, N> operator+(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_add_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N>& operator+=(const FixedVector<double, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_add_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator-() const {
      FixedVector<double, N, true> result;
      const __m128d sign_mask = _mm_set1_pd(-0.0);
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_xor_pd(data[i], sign_mask);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator-(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_sub_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator-=(const FixedVector<double, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_sub_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator*(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_mul_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator*=(const FixedVector<double, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_mul_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true> operator/(const FixedVector<double, N, true>& v) const {
      FixedVector<double, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_div_pd(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<double, N, true>& operator/=(const FixedVector<double, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_div_pd(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm_cmplt_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm_cmplt_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm_cmple_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm_cmple_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm_cmpeq_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm_cmpeq_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm_cmpge_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm_cmpge_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<double, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_MASK_LOOP(i) {
        result.data[i] = CollapseDoubles(_mm_cmpgt_pd(data[2*i+0], v.data[2*i+0]),
                                         _mm_cmpgt_pd(data[2*i+1], v.data[2*i+1]));
      }
      return result;
    }

    SYRAH_FORCEINLINE double MaxElement() const {
      __m128d tmpMax = data[0];
      for (int i = 1; i < N/2; i++) {
        tmpMax = _mm_max_pd(data[i], tmpMax);
      }

      // Now we've got to reduce the 4-wide version into a single double.
      return syrah_mm_hmax_pd(tmpMax);
    }

    SYRAH_FORCEINLINE double MinElement() const {
      __m128d tmpMin = data[0];
      for (int i = 1; i < N/2; i++) {
        tmpMin = _mm_min_pd(data[i], tmpMin);
      }

      // Now we've got to reduce the 4-wide version into a single double.
      return syrah_mm_hmin_pd(tmpMin);
    }

    SYRAH_FORCEINLINE double foldMax() const { return MaxElement(); }
    SYRAH_FORCEINLINE double foldMin() const { return MinElement(); }

    SYRAH_FORCEINLINE double foldSum() const {
      __m128d tmp_sum = _mm_setzero_pd();
      SYRAH_SSE_LOOP(i) {
        tmp_sum = _mm_add_pd(tmp_sum, data[i]);
      }
      // Have [a, b] want a+b
      __m128d ab = _mm_hadd_pd(tmp_sum, _mm_setzero_pd()); // [a+b, c+d, 0, 0]
      return _mm_cvtsd_f64(ab);
    }

    SYRAH_FORCEINLINE double foldProd() const {
      __m128d tmp_prod = _mm_set1_pd(1.0);
      SYRAH_SSE_LOOP(i) {
        tmp_prod = _mm_mul_pd(tmp_prod, data[i]);
      }
      // Have [a, b] want a*b

      // temp1 = [a, b] * [b, b] = [a * b, b^2]
      __m128d ab = _mm_mul_pd(tmp_prod, _mm_unpackhi_pd(tmp_prod, tmp_prod));
      return _mm_cvtsd_f64(ab);
    }

    __m128d data[N/2];
  };

  // TODO(boulos): scalar BINOP vector and vector BINOP scalar

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> max(const FixedVector<double, N, true>& v1, const FixedVector<double, N, true>& v2) {
    FixedVector<double, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_max_pd(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> min(const FixedVector<double, N, true>& v1, const FixedVector<double, N, true>& v2) {
    FixedVector<double, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_min_pd(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> sqrt(const FixedVector<double, N, true>& v1) {
    FixedVector<double, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_sqrt_pd(v1.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> abs(const FixedVector<double, N, true>& v1) {
    FixedVector<double, N, true> result;
    // -0 is 0x8000000000000000
    __m128d sign_mask = _mm_set1_pd(-0.0);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_andnot_pd(sign_mask, v1.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> madd(const FixedVector<double, N, true>& a,
                                           const FixedVector<double, N, true>& b,
                                           const FixedVector<double, N, true>& c) {
    FixedVector<double, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_add_pd(_mm_mul_pd(a.data[i], b.data[i]), c.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> trunc(const FixedVector<double, N, true>& a) {
    FixedVector<double, N, true> result;
    SYRAH_SSE_LOOP(i) {
#if defined(__SSE4_1__)
      result.data[i] = _mm_round_pd(a.data[i], 3 /* round towards 0 = truncate */);
#else
      // NOTE(boulos): there's cvttpd and cvtpd (the extra t is for truncate)
      result.data[i] = _mm_cvtepi32_pd(_mm_cvttpd_epi32(a.data[i]));
#endif
    }
    return result;
  }

  // result = round_to_float(a) (using current rounding mode)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> rint(const FixedVector<double, N, true>& a) {
    FixedVector<double, N, true> result;
    SYRAH_SSE_LOOP(i) {
#if defined(__SSE4_1__)
      result.data[i] = _mm_round_pd(a.data[i], 4 /* round using rounding mode */);
#else
      // as above, cvtps is for "current rounding mode"
      result.data[i] = _mm_cvtepi32_pd(_mm_cvtpd_epi32(a.data[i]));
#endif
    }
    return result;
  }

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N, true> select(const FixedVector<double, N, true>& a,
                                             const FixedVector<double, N, true>& b,
                                             const FixedVectorMask<N>& mask) {
    FixedVector<double, N, true> result;
    SYRAH_SSE_LOOP(i) {
      __m128d mask_double = ExpandToDouble(mask.data[i/2], i & 1);
      result.data[i] = syrah_blendv_pd(a.data[i], b.data[i], mask_double);
    }
    return result;
  }
#undef SYRAH_MASK_LOOP
#undef SYRAH_SSE_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_SSE_DOUBLE_H_
