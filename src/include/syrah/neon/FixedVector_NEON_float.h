#ifndef _SYRAH_FIXED_VECTOR_NEON_FLOAT_H_
#define _SYRAH_FIXED_VECTOR_NEON_FLOAT_H_

#include "NEON_intrin.h"

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(16) FixedVector<float, N, true> {
    public:
#define SYRAH_NEON_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)

    SYRAH_FORCEINLINE FixedVector() {}

    SYRAH_FORCEINLINE FixedVector(const float value) {
      load(value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values) {
      load(values);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, bool /* aligned */) {
      load_aligned(values);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, const FixedVectorMask<N>& mask,
                                  const float default_value) {
      load(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, const FixedVectorMask<N>& mask,
                                  const float default_value, bool /*aligned */) {
      load_aligned(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* base, const FixedVector<int, N, true>& offsets, const int scale) {
      gather(base, offsets, scale);
    }

    SYRAH_FORCEINLINE FixedVector(const float* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      gather(base, offsets, scale, mask);
    }

    // Have to wait until <int> is ready.
    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<int, N> &v);

    SYRAH_FORCEINLINE FixedVector(const FixedVector<float, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_NEON_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    static SYRAH_FORCEINLINE FixedVector<float, N, true> reinterpret(const FixedVector<int, N, true>& v);

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator=(const FixedVector<float, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    static SYRAH_FORCEINLINE FixedVector<float, N, true> Zero() {
      FixedVector<float, N, true> result;
      uint32x4_t zero_int;
      zero_int = veorq_u32(zero_int, zero_int);
      float32x4_t zero_f = vreinterpretq_f32_u32(zero_int);
      SYRAH_NEON_LOOP(i) {
        result.data[i] = zero_f;
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
      float32x4_t splat = vdupq_n_f32(value);
      SYRAH_NEON_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const float* values) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vld1q_f32((const float32_t*)&values[4*i]);
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const float* values) {
      load(values);
    }

    SYRAH_FORCEINLINE void load(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
      float32x4_t default_data = vdupq_n_f32(default_value);
      SYRAH_NEON_LOOP(i) {
        float32x4_t loaded_data = vld1q_f32((const float32_t*)&(values[4*i]));
        data[i] = vblsq_f32(mask.data[i], loaded_data, default_data);
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
      load(values, mask, default_value);
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
      SYRAH_NEON_LOOP(i) {
        vst1q_f32((float32_t*)&(dst[4*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst) const {
      store(dst);
    }

    SYRAH_FORCEINLINE void store_aligned_stream(float* dst) const {
      store(dst);
    }

    SYRAH_FORCEINLINE void store(float* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_NEON_LOOP(i) {
        float32x4_t cur_val = vld1q_f32((const float32_t*)&(dst[4*i]));
        vst1q_f32((float32_t*)&(dst[4*i]), vbslq_f32(mask.data[i], data[i], cur_val));
      }
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst, const FixedVectorMask<N>& mask) const {
      store(dst, mask);
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
      SYRAH_NEON_LOOP(i) {
        data[i] = vbslq_f32(mask.data[i], v.data[i], data[i]);
      }
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vaddq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator+=(const FixedVector<float, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vaddq_f32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator-() const {
      FixedVector<float, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vnegq_f32(data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vsubq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator-=(const FixedVector<float, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vsubq_f32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vmulq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator*=(const FixedVector<float, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vmulq_f32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vmulq_f32(data[i], syrah_rcp(v.data[i]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator/=(const FixedVector<float, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vmulq_f32(data[i], syrah_rcp(v.data[i]));
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcltq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcleq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vceqq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcgeq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcgtq_f32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE float MaxElement() const {
      float32x4_t tmpMax = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMax = vmaxq_f32(data[i], tmpMax);
      }

      float32x2_t lo_part = vget_low_f32(tmpMax);
      float32x2_t hi_part = vget_high_f32(tmpMax);
      float32x2_t fold_max = vpmax_f32(lo_part, hi_part);
      fold_max = vpmax_f32(fold_max, fold_max);
      return vget_lane_f32(fold_max, 0);
    }

    SYRAH_FORCEINLINE float MinElement() const {
      float32x4_t tmpMin = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMin = vminq_f32(data[i], tmpMin);
      }

      float32x2_t lo_part = vget_low_f32(tmpMin);
      float32x2_t hi_part = vget_high_f32(tmpMin);
      float32x2_t fold_min = vpmin_f32(lo_part, hi_part);
      fold_min = vpmin_f32(fold_min, fold_min);
      return vget_lane_f32(fold_min, 0);
    }

    SYRAH_FORCEINLINE float foldMax() const { return MaxElement(); }
    SYRAH_FORCEINLINE float foldMin() const { return MinElement(); }

    SYRAH_FORCEINLINE float foldSum() const {
      uint32x4_t zero_int;
      zero_int = veorq_u32(zero_int, zero_int);
      float32x4_t zero_f = vreinterpretq_f32_u32(zero_int);
      float32x4_t tmp_sum = zero_f;
      SYRAH_NEON_LOOP(i) {
        tmp_sum = vaddq_f32(tmp_sum, data[i]);
      }

      // [a, b]
      float32x2_t lo_part = vget_low_f32(tmp_sum);
      // [c, d]
      float32x2_t hi_part = vget_high_f32(tmp_sum);
      // [a+c, b+d]
      float32x2_t fold_sum = vpadd_f32(lo_part, hi_part);
      // [(a+c)+(b+d), (a+c)+(b+d)]
      fold_sum = vpadd_f32(fold_sum, fold_sum);
      return vget_lane_f32(fold_sum, 0);
    }

    SYRAH_FORCEINLINE float foldProd() const {
      float32x4_t tmp_prod = vdupq_n_f32(1.f);
      SYRAH_NEON_LOOP(i) {
        tmp_prod = vmulq_f32(tmp_prod, data[i]);
      }

      // [a, b, c, d]
      float32x2_t ab = vget_low_f32(tmp_prod);
      float32x2_t cd = vget_high_f32(tmp_prod);
      float32x2_t ac_bd = vmul_f32(ab, cd);
      float32x2_t bd = vdup_lane_f32(ac_bd, 1);
      float32x2_t acbd = vmul_f32(ac_bd, bd);
      return vget_lane_f32(acbd, 0);
    }


    float32x4_t data[N/4];
  };

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vaddq_f32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vaddq_f32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vsubq_f32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vsubq_f32(v.data[i], splat);
    }
    return result;
  }

  // NOTE(boulos): FP mul is associative so we can use
  // vmulq_n_f32(vec, scalar) for both scalar * vec and vec * scalar.
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmulq_n_f32(v.data[i], s);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmulq_n_f32(v.data[i], s);
    }
    return result;
  }

  // QUESTION(boulos): Is it better to use vmulq_n_f32 here too?
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const float& s, const FixedVector<float, N, true>& v) {
    FixedVector<float, N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmulq_f32(splat, syrah_rcp(v.data[i]));
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const FixedVector<float, N, true>& v, const float& s) {
    FixedVector<float, N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    float32x4_t inv_splat = syrah_rcp(splat);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmulq_f32(v.data[i], inv_splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcltq_f32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcltq_f32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcleq_f32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcleq_f32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vceqq_f32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vceqq_f32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgeq_f32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgeq_f32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const float& s, const FixedVector<float, N, true>& v) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgtq_f32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const FixedVector<float, N, true>& v, const float& s) {
    FixedVectorMask<N> result;
    const float32x4_t splat = vdupq_n_f32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgtq_f32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const float& s, const FixedVector<float, N, true>& v) {
    return !(operator==(s,v));
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<float, N, true>& v, const float& s) {
    return !(operator==(v, s));
  }


  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> max(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmaxq_f32(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> min(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vminq_f32(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> sqrt(const FixedVector<float, N, true>& v) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      float32x4_t inv_sqrt_v = vrsqrteq_f32(v.data[i]);
      float32x4_t step_val = vrsqrtsq_f32(v.data[i], inv_sqrt_v);
      float32x4_t true_inv_sqrt = vmulq_f32(inv_sqrt_v, step_val);
      result.data[i] = vmulq_f32(v.data[i], true_inv_sqrt);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> madd(const FixedVector<float, N, true>& a,
                                                    const FixedVector<float, N, true>& b,
                                                    const FixedVector<float, N, true>& c) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      //result.data[i] = vmlaq_f32(a.data[i], b.data[i], c.data[i]);
      //result.data[i] = vmlaq_f32(b.data[i], c.data[i], a.data[i]);
      result.data[i] = vaddq_f32(c.data[i], vmulq_f32(a.data[i], b.data[i]));
    }
    return result;
  }

  // result = (float)((int)a)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> trunc(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcvtq_f32_s32(vcvtq_s32_f32(a.data[i]));
    }
    return result;
  }

  // result = round_to_float(a) (using current rounding mode)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> rint(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcvtq_f32_s32(vcvtq_s32_f32(a.data[i]));
    }
    return result;
  }

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> select(const FixedVector<float, N, true>& a,
                                                       const FixedVector<float, N, true>& b,
                                                       const FixedVectorMask<N>& mask) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vbslq_f32(mask.data[i], a.data[i], b.data[i]);
    }
    return result;
  }
#undef SYRAH_NEON_LOOP
} // end namespace syrah

#endif //_SYRAH_FIXED_VECTOR_NEON_FLOAT_H_
