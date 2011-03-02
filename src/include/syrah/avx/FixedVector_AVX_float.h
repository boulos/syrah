#ifndef _SYRAH_FIXED_VECTOR_AVX_FLOAT_H_
#define _SYRAH_FIXED_VECTOR_AVX_FLOAT_H_

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(32) FixedVector<float, N, true> {
  public:
#define SYRAH_AVX_LOOP(index) SYRAH_UNROLL(N/8) \
      for (int index = 0; index < N/8; index++)

  SYRAH_FORCEINLINE FixedVector() {}

  SYRAH_FORCEINLINE FixedVector(const float value) {
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
    SYRAH_AVX_LOOP(i) {
      data[i] = _mm256_set_ps(v[8*i + 7], v[8*i + 6], v[8*i + 5], v[8*i + 4], v[8*i + 3], v[8*i+2], v[8*i+1], v[8*i+0]);
    }
  }

  // Have to wait until <int> is ready.
  SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<int, N> &v);

  SYRAH_FORCEINLINE FixedVector(const FixedVector<float, N, true>& v) {
    // TODO(boulos): Optimize this
    SYRAH_AVX_LOOP(i) {
      data[i] = v.data[i];
    }
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true>& operator=(const FixedVector<float, N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      data[i] = v.data[i];
    }
    return *this;
  }

  static SYRAH_FORCEINLINE FixedVector<float, N, true> reinterpret(const FixedVector<int, N, true>& v);

  static SYRAH_FORCEINLINE FixedVector<float, N, true> Zero() {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_setzero_ps();
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
    __m256 splat = _mm256_set1_ps(value);
    SYRAH_AVX_LOOP(i) {
      data[i] = splat;
    }
  }

  SYRAH_FORCEINLINE void load(const float* values) {
    bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 32);
    if (sse_aligned) {
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_load_ps(&values[8*i]);
      }
    } else {
      // TODO(boulos): If N is long enough, this could probably be
      // improved.
      SYRAH_AVX_LOOP(i) {
        data[i] = _mm256_loadu_ps(&values[8*i]);
      }
    }
  }

  SYRAH_FORCEINLINE void load_aligned(const float* values) {
    SYRAH_AVX_LOOP(i) {
      data[i] = _mm256_load_ps(&values[8*i]);
    }
  }

  SYRAH_FORCEINLINE void load(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
    bool sse_aligned = SYRAH_IS_ALIGNED_POW2(values, 32);
    __m256 default_data = _mm256_set1_ps(default_value);
    if (sse_aligned) {
      SYRAH_AVX_LOOP(i) {
        __m256 loaded_data = _mm256_load_ps(&(values[8*i]));
        data[i] = _mm256_blendv_ps(default_data, loaded_data, mask.data[i]);
      }
    } else {
      // TODO(boulos): If N is long enough, this could probably be
      // improved.
      SYRAH_AVX_LOOP(i) {
        __m256 loaded_data = _mm256_loadu_ps(&(values[8*i]));
        data[i] = _mm256_blendv_ps(default_data, loaded_data, mask.data[i]);
      }
    }
  }

  SYRAH_FORCEINLINE void load_aligned(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
    __m256 default_data = _mm256_set1_ps(default_value);
    SYRAH_AVX_LOOP(i) {
      __m256 loaded_data = _mm256_load_ps(&(values[8*i]));
      data[i] = _mm256_blendv_ps(default_value, loaded_data, mask.data[i]);
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
    bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 32);
    if (sse_aligned) {
      SYRAH_AVX_LOOP(i) {
        _mm256_store_ps(&(dst[8*i]), data[i]);
      }
    } else {
      SYRAH_AVX_LOOP(i) {
        _mm256_storeu_ps(&(dst[8*i]), data[i]);
      }
    }
  }

  SYRAH_FORCEINLINE void store_aligned(float* dst) const {
    SYRAH_AVX_LOOP(i) {
      _mm256_store_ps(&(dst[8*i]), data[i]);
    }
  }

  SYRAH_FORCEINLINE void store_aligned_stream(float* dst) const {
    SYRAH_AVX_LOOP(i) {
      _mm256_stream_ps(&(dst[8*i]), data[i]);
    }
  }

  SYRAH_FORCEINLINE void store(float* dst, const FixedVectorMask<N>& mask) const {
    bool sse_aligned = SYRAH_IS_ALIGNED_POW2(dst, 32);
    if (sse_aligned) {
      SYRAH_AVX_LOOP(i) {
        __m256 cur_val = _mm256_load_ps(&(dst[8*i]));
        _mm256_store_ps(&(dst[8*i]), _mm256_blendv_ps(cur_val, data[i], mask.data[i]));
      }
    } else {
      SYRAH_AVX_LOOP(i) {
        __m256 cur_val = _mm256_loadu_ps(&(dst[8*i]));
        _mm256_storeu_ps(&(dst[8*i]), _mm256_blendv_ps(cur_val, data[i], mask.data[i]));
      }
    }
  }

  SYRAH_FORCEINLINE void store_aligned(float* dst, const FixedVectorMask<N>& mask) const {
    SYRAH_AVX_LOOP(i) {
      __m256 cur_val = _mm256_load_ps(&(dst[8*i]));
      _mm256_store_ps(&(dst[8*i]), _mm256_blendv_ps(cur_val, data[i], mask.data[i]));
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
    SYRAH_AVX_LOOP(i) {
      data[i] = _mm256_blendv_ps(data[i], v.data[i], mask.data[i]);
    }
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const FixedVector<float, N, true>& v) const {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_add_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true>& operator+=(const FixedVector<float, N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      data[i] = _mm256_add_ps(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-() const {
    FixedVector<float, N, true> result;
    const __m256 sign_mask = _mm256_set1_ps(-0.f);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_xor_ps(data[i], sign_mask);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const FixedVector<float, N, true>& v) const {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_sub_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true>& operator-=(const FixedVector<float, N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      data[i] = _mm256_sub_ps(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const FixedVector<float, N, true>& v) const {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_mul_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true>& operator*=(const FixedVector<float, N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      data[i] = _mm256_mul_ps(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const FixedVector<float, N, true>& v) const {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_div_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<float, N, true>& operator/=(const FixedVector<float, N, true>& v) {
    SYRAH_AVX_LOOP(i) {
      data[i] = _mm256_div_ps(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<float, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmplt_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<float, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmple_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<float, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpeq_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<float, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpneq_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<float, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpge_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<float, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpgt_ps(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE float MaxElement() const {
    __m256 tmpMax = data[0];
    for (int i = 1; i < N/8; i++) {
      tmpMax = _mm256_max_ps(data[i], tmpMax);
    }

    // Now we've got to reduce the 8-wide version into a single float.
    return syrah_mm256_hmax_ps(tmpMax);
  }

  SYRAH_FORCEINLINE float MinElement() const {
    __m256 tmpMin = data[0];
    for (int i = 1; i < N/8; i++) {
      tmpMin = _mm256_min_ps(data[i], tmpMin);
    }

    // Now we've got to reduce the 8-wide version into a single float.
    return syrah_mm256_hmin_ps(tmpMin);
  }

  SYRAH_FORCEINLINE float foldMax() const { return MaxElement(); }
  SYRAH_FORCEINLINE float foldMin() const { return MinElement(); }

  SYRAH_FORCEINLINE float foldSum() const {
    __m256 tmp_sum = _mm256_setzero_ps();
    SYRAH_AVX_LOOP(i) {
      tmp_sum = _mm256_add_ps(tmp_sum, data[i]);
    }

    // ab, cd, 0, 0, ef, gh, 0, 0
    __m256 ab_cd_0_0_ef_gh_0_0 = _mm256_hadd_ps(tmp_sum, _mm256_setzero_ps());
    __m256 abcd_0_0_0_efgh_0_0_0 = _mm256_hadd_ps(ab_cd_0_0_ef_gh_0_0, _mm256_setzero_ps());
    // get efgh in the 0-idx slot
    __m256 efgh = _mm256_permute2f128_ps(abcd_0_0_0_efgh_0_0_0, abcd_0_0_0_efgh_0_0_0, 0x1);
    __m256 final_sum = _mm256_add_ps(abcd_0_0_0_efgh_0_0_0, efgh);
    return _mm_cvtss_f32(_mm256_castps256_ps128(final_sum));
  }

  SYRAH_FORCEINLINE float foldProd() const {
    __m256 tmp_prod = _mm256_set1_ps(1.f);
    SYRAH_AVX_LOOP(i) {
      tmp_prod = _mm256_mul_ps(tmp_prod, data[i]);
    }
    // Have [a, b, c, d, e, f, g, h] want a * b * c * d * e * ... * h
    // temp1 = [a, b, c, d, e, f, g, h] * [e, f, g, h, X, X, X, X]
    //       = [ae, bf, cg, dh, X, X, X, X]
    __m256 temp1 = _mm256_mul_ps(tmp_prod, _mm256_permute2f128_ps(tmp_prod, tmp_prod, 0x1));
    // temp2 = [cg, dh, cg, dh, X, X, X, X]
    __m256 temp2 = _mm256_shuffle_ps(temp1, temp1, _MM_SHUFFLE(3, 2, 3, 2));
    // temp3 = [aecg, bfdh, cg^2, dh^2, X, X, X, X]
    __m256 temp3 = _mm256_mul_ps(temp1, temp2);
    __m256 bfdh = _mm256_shuffle_ps(temp3, temp3, _MM_SHUFFLE(1, 1, 1, 1));
    __m256 final_prod = _mm256_mul_ps(temp3, bfdh);
    return _mm_cvtss_f32(_mm256_castps256_ps128(final_prod));
  }

  __m256 data[N/8];
  };


  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_add_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_add_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_sub_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_sub_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_mul_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_mul_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_div_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_div_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmplt_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmplt_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmple_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmple_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpeq_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpeq_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpge_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpge_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpgt_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpgt_ps(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpneq_ps(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const __m256 splat = _mm256_set1_ps(s);
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_cmpneq_ps(v.data[i], splat);
    }
    return result;
  }


  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> max(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_max_ps(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> min(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_min_ps(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> sqrt(const FixedVector<float, N, true>& v1) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_sqrt_ps(v1.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> madd(const FixedVector<float, N, true>& a,
                                                     const FixedVector<float, N, true>& b,
                                                     const FixedVector<float, N, true>& c) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_add_ps(_mm256_mul_ps(a.data[i], b.data[i]), c.data[i]);
    }
    return result;
  }

  // result = (float)((int)a)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> trunc(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_round_ps(a.data[i], 3 /* round towards 0 = truncate */);
    }
    return result;
  }

  // result = round_to_float(a) (using current rounding mode)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> rint(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_round_ps(a.data[i], 4 /* round using rounding mode */);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> floor(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_floor_ps(a.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> ceil(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_ceil_ps(a.data[i]);
    }
    return result;
  }

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> select(const FixedVector<float, N, true>& a,
                                                       const FixedVector<float, N, true>& b,
                                                       const FixedVectorMask<N>& mask) {
    FixedVector<float, N, true> result;
    SYRAH_AVX_LOOP(i) {
      result.data[i] = _mm256_blendv_ps(b.data[i], a.data[i], mask.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> reverse(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    // To reverse a N-wide vector, we first have to reverse the data
    // array and then the bits themselves.
    SYRAH_AVX_LOOP(i) {
      __m256 other = a.data[N/8 - i - 1];
      // Can't shuffle the whole thing with AVX, need to get the first half and the second half
      __m128 lo_part = _mm256_extractf128_ps(other, 0);
      __m128 hi_part = _mm256_extractf128_ps(other, 1);

      // Reverse the parts
      lo_part = _mm_shuffle_ps(lo_part, lo_part, _MM_SHUFFLE(0, 1, 2, 3));
      hi_part = _mm_shuffle_ps(hi_part, hi_part, _MM_SHUFFLE(0, 1, 2, 3));

      // Put it back together but in reverse part order (reversed_hi, reversed_lo)
      __m256 reversed = _mm256_insertf128_ps(_mm256_castps128_ps256(hi_part), lo_part, 1);
      result.data[i] = reversed;
    }
    return result;
  }


#undef SYRAH_AVX_LOOP
} // end namespace syrah

#endif //_SYRAH_FIXED_VECTOR_AVX_FLOAT_H_
