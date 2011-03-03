#ifndef _SYRAH_FIXED_VECTOR_SSE_FLOAT_H_
#define _SYRAH_FIXED_VECTOR_SSE_FLOAT_H_

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(16) FixedVector<float, N, true> {
    public:
#define SYRAH_SSE_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)

    SYRAH_FORCEINLINE FixedVector() {}

    SYRAH_FORCEINLINE FixedVector(float value) {
      load(value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values) {
      load(values);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, bool aligned) {
      load_aligned(values);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, const FixedVectorMask<N>& mask,
                       const float default_value) {
      load(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, const FixedVectorMask<N>& mask,
                                 const float default_value, bool aligned) {
      load_aligned(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* base, const FixedVector<int, N, true>& offsets, const int scale) {
      gather(base, offsets, scale);
    }

    SYRAH_FORCEINLINE FixedVector(const float* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      gather(base, offsets, scale, mask);
    }

    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<double, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_set_ps(v[4*i + 3], v[4*i+2], v[4*i+1], v[4*i+0]);
      }
    }

    // Have to wait until <int> is ready.
    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<int, N> &v);

    SYRAH_FORCEINLINE FixedVector(const FixedVector<float, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator=(const FixedVector<float, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    static SYRAH_FORCEINLINE FixedVector<float, N, true> reinterpret(const FixedVector<int, N, true>& v);

    static SYRAH_FORCEINLINE FixedVector<float, N, true> Zero() {
      FixedVector<float, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_setzero_ps();
      }
      return result;
    }


    SYRAH_FORCEINLINE const float& operator[](int i) const {
      return ((float*)data)[i];
    }

    SYRAH_FORCEINLINE float& operator[](int i) {
      return ((float*)data)[i];
    }

    SYRAH_FORCEINLINE void load(const float value) {
      __m128 splat = _mm_set1_ps(value);
      SYRAH_SSE_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const float* values) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          data[i] = _mm_load_ps(&values[4*i]);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_SSE_LOOP(i) {
          data[i] = _mm_loadu_ps(&values[4*i]);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const float* values) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_load_ps(&values[4*i]);
      }
    }

    SYRAH_FORCEINLINE void load(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 16);
      __m128 default_data = _mm_set1_ps(default_value);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          __m128 loaded_data = _mm_load_ps(&(values[4*i]));
          data[i] = syrah_blendv_ps(loaded_data, default_data, mask.data[i]);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_SSE_LOOP(i) {
          __m128 loaded_data = _mm_loadu_ps(&(values[4*i]));
          data[i] = syrah_blendv_ps(loaded_data, default_data, mask.data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
      __m128 default_data = _mm_set1_ps(default_value);
      SYRAH_SSE_LOOP(i) {
        __m128 loaded_data = _mm_load_ps(&(values[4*i]));
        data[i] = syrah_blendv_ps(loaded_data, default_data, mask.data[i]);
      }
    }

    SYRAH_FORCEINLINE void gather(const float* base, const FixedVector<int, N, true>& offsets, const int scale) {
      float* float_data = reinterpret_cast<float*>(data);
      // TODO(boulos): Consider doing all the indexing ops with
      // vector? Seems pointless.
      for (int i = 0; i < N; i++) {
        const float* addr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
        float_data[i] = *addr;
      }
    }

    // QUESTION(boulos): Constant scale only?
    SYRAH_FORCEINLINE void gather(const float* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      float* float_data = reinterpret_cast<float*>(data);
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          const float* addr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
          float_data[i] = *addr;
        }
      }
    }

    SYRAH_FORCEINLINE void store(float* dst) const {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          _mm_store_ps(&(dst[4*i]), data[i]);
        }
      } else {
        SYRAH_SSE_LOOP(i) {
          _mm_storeu_ps(&(dst[4*i]), data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst) const {
      SYRAH_SSE_LOOP(i) {
        _mm_store_ps(&(dst[4*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store_aligned_stream(float* dst) const {
      SYRAH_SSE_LOOP(i) {
        _mm_stream_ps(&(dst[4*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store(float* dst, const FixedVectorMask<N>& mask) const {
      bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 16);
      if (sse_aligned) {
        SYRAH_SSE_LOOP(i) {
          __m128 cur_val = _mm_load_ps(&(dst[4*i]));
          _mm_store_ps(&(dst[4*i]), syrah_blendv_ps(data[i], cur_val, mask.data[i]));
        }
      } else {
        SYRAH_SSE_LOOP(i) {
          __m128 cur_val = _mm_loadu_ps(&(dst[4*i]));
          _mm_storeu_ps(&(dst[4*i]), syrah_blendv_ps(data[i], cur_val, mask.data[i]));
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_SSE_LOOP(i) {
        __m128 cur_val = _mm_load_ps(&(dst[4*i]));
        _mm_store_ps(&(dst[4*i]), syrah_blendv_ps(data[i], cur_val, mask.data[i]));
      }
    }

    SYRAH_FORCEINLINE void scatter(float* base, const FixedVector<int, N, true>& offsets, const int scale) const {
      const float* float_data = reinterpret_cast<const float*>(data);
      // TODO(boulos): Consider doing all the indexing ops with
      // vector? Seems pointless.
      for (int i = 0; i < N; i++) {
        float* addr = reinterpret_cast<float*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
        *addr = float_data[i];
      }
    }

    // QUESTION(boulos): Constant scale only?
    SYRAH_FORCEINLINE void scatter(float* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) const {
      const float* float_data = reinterpret_cast<const float*>(data);
      for (int i = 0; i < N; i++) {
        if (mask.get(i)) {
          float* addr = reinterpret_cast<float*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
          *addr = float_data[i];
        }
      }
    }

    SYRAH_FORCEINLINE void merge(const FixedVector<float, N, true>& v,
                                const FixedVectorMask<N, true>& mask) {
      SYRAH_SSE_LOOP(i) {
        data[i] = syrah_blendv_ps(v.data[i], data[i], mask.data[i]);
      }
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_add_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator+=(const FixedVector<float, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_add_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator-() const {
      FixedVector<float, N, true> result;
      // NOTE(boulos): xor is 1 cycle latency while _sub_ps is 3
      // cycles.
      const __m128 sign_mask = _mm_set1_ps(-0.f);
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_xor_ps(data[i], sign_mask);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_sub_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator-=(const FixedVector<float, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_sub_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_mul_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator*=(const FixedVector<float, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_mul_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_div_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator/=(const FixedVector<float, N, true>& v) {
      SYRAH_SSE_LOOP(i) {
        data[i] = _mm_div_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_cmplt_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_cmple_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_cmpeq_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_cmpge_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_cmpgt_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_SSE_LOOP(i) {
        result.data[i] = _mm_cmpneq_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE float MaxElement() const {
      __m128 tmpMax = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMax = _mm_max_ps(data[i], tmpMax);
      }

      // Now we've got to reduce the 4-wide version into a single float.
      return syrah_mm_hmax_ps(tmpMax);
    }

    SYRAH_FORCEINLINE float MinElement() const {
      __m128 tmpMin = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMin = _mm_min_ps(data[i], tmpMin);
      }

      // Now we've got to reduce the 4-wide version into a single float.
      return syrah_mm_hmin_ps(tmpMin);
    }

    SYRAH_FORCEINLINE float foldMax() const { return MaxElement(); }
    SYRAH_FORCEINLINE float foldMin() const { return MinElement(); }

    SYRAH_FORCEINLINE float foldSum() const {
      __m128 tmp_sum = _mm_setzero_ps();
      SYRAH_SSE_LOOP(i) {
        tmp_sum = _mm_add_ps(tmp_sum, data[i]);
      }
      // Have [a, b, c, d] want a + b + c + d
      __m128 ab_cd_0_0 = _mm_hadd_ps(tmp_sum, _mm_setzero_ps()); // [a+b, c+d, 0, 0]
      __m128 abcd = _mm_hadd_ps(ab_cd_0_0, _mm_setzero_ps()); // [a+b+c+d x2, 0, 0]
      return _mm_cvtss_f32(abcd);
    }

    SYRAH_FORCEINLINE float foldProd() const {
      __m128 tmp_prod = _mm_set1_ps(1.f);
      SYRAH_SSE_LOOP(i) {
        tmp_prod = _mm_mul_ps(tmp_prod, data[i]);
      }
      // Have [a, b, c, d] want a * b * c * d

      // temp1 = [a, b, c, d] * [b, b, d, d] = [a * b, b^2, c*d, d^2]
      __m128 temp1 = _mm_mul_ps(tmp_prod, _mm_movehdup_ps(tmp_prod));
      // Get c*d out
      __m128 cd = _mm_shuffle_ps(temp1, temp1, _MM_SHUFFLE(2, 2, 2, 2));
      __m128 abcd = _mm_mul_ps(temp1, cd);
      return _mm_cvtss_f32(abcd);
    }

    __m128 data[N/4];
  };

  // Scalar BINOP Vector (because of lame C++ template deduction
  // implicit type conversion rules and because it should be more
  // efficient for use cases where a constant is multiplied by say a
  // 16-vec. Instead of 4 SSE registers to represent the scalar it'll
  // definitely only use 1)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_add_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_add_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_sub_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_sub_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_mul_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_mul_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_div_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_div_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmplt_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmplt_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmple_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmple_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpeq_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpeq_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpge_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpge_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpgt_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpgt_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpneq_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_cmpneq_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> max(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_max_ps(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> max(const float& s, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_max_ps(splat, v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> max(const FixedVector<float, N, true>& v2, const float& s) {
    FixedVector<float, N, true> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_max_ps(v2.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> min(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_min_ps(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> min(const float& s, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_min_ps(s, v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> min(const FixedVector<float, N, true>& v2, const float& s) {
    FixedVector<float, N, true> result;
    const __m128 splat = _mm_set1_ps(s);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_min_ps(v2.data[i], s);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> sqrt(const FixedVector<float, N, true>& v1) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_sqrt_ps(v1.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> abs(const FixedVector<float, N, true>& v1) {
    FixedVector<float, N, true> result;
    // -0 => 0x80000000.
    __m128 sign_mask = _mm_set1_ps(-0.0f);
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_andnot_ps(sign_mask, v1.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> madd(const FixedVector<float, N, true>& a,
                                                    const FixedVector<float, N, true>& b,
                                                    const FixedVector<float, N, true>& c) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_add_ps(_mm_mul_ps(a.data[i], b.data[i]), c.data[i]);
    }
    return result;
  }

  // result = (float)((int)a)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> trunc(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
#if defined(__SSE4_1__) && !defined(__clang__)
      result.data[i] = _mm_round_ps(a.data[i], 3 /* round towards 0 = truncate */);
#else
      // NOTE(boulos): there's cvttps and cvtps (the extra t is for truncate)
      result.data[i] = _mm_cvtepi32_ps(_mm_cvttps_epi32(a.data[i]));
#endif
    }
    return result;
  }

  // result = round_to_float(a) (using current rounding mode)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> rint(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
#if defined(__SSE4_1__)
      result.data[i] = _mm_round_ps(a.data[i], 4 /* round using rounding mode */);
#else
      // as above, cvtps is for "current rounding mode"
      result.data[i] = _mm_cvtepi32_ps(_mm_cvtps_epi32(a.data[i]));
#endif
    }
    return result;
  }

  // Makes sense to override floor and ceil for SSE4.1.
#if defined(__SSE4_1__)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> floor(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_floor_ps(a.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> ceil(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_ceil_ps(a.data[i]);
    }
    return result;
  }
#endif

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> select(const FixedVector<float, N, true>& a,
                                                       const FixedVector<float, N, true>& b,
                                                       const FixedVectorMask<N>& mask) {
    FixedVector<float, N, true> result;
    SYRAH_SSE_LOOP(i) {
      result.data[i] = syrah_blendv_ps(a.data[i], b.data[i], mask.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> reverse(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    // To reverse a N-wide vector, we first have to reverse the data
    // array and then the bits themselves.
    SYRAH_SSE_LOOP(i) {
      result.data[i] = _mm_shuffle_ps(a.data[N/4 - i - 1], a.data[N/4 - i - 1], _MM_SHUFFLE(0, 1, 2, 3));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> splat(const FixedVector<float, N, true>& a, int which) {
    FixedVector<float, N, true> result;
    // To splat from an N-wide vector, we first have to figure out
    // which sse value to splat from.
    int which_sse = which >> 2;
    int which_elem = which & 3;
    switch (which_elem) {
    case 0: SYRAH_SSE_LOOP(i) { result.data[i] = _mm_shuffle_ps(a.data[which_sse], a.data[which_sse], _MM_SHUFFLE(0, 0, 0, 0)); } break;
    case 1: SYRAH_SSE_LOOP(i) { result.data[i] = _mm_shuffle_ps(a.data[which_sse], a.data[which_sse], _MM_SHUFFLE(1, 1, 1, 1)); } break;
    case 2: SYRAH_SSE_LOOP(i) { result.data[i] = _mm_shuffle_ps(a.data[which_sse], a.data[which_sse], _MM_SHUFFLE(2, 2, 2, 2)); } break;
    default: SYRAH_SSE_LOOP(i) { result.data[i] = _mm_shuffle_ps(a.data[which_sse], a.data[which_sse], _MM_SHUFFLE(3, 3, 3, 3)); } break;
    }
    return result;
  }


#undef SYRAH_SSE_LOOP
} // end namespace syrah

#endif //_SYRAH_FIXED_VECTOR_SSE_FLOAT_H_
