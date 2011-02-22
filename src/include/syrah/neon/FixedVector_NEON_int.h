#ifndef _SYRAH_FIXED_VECTOR_NEON_INT_H_
#define _SYRAH_FIXED_VECTOR_NEON_INT_H_

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(16) FixedVector<int, N, true> {
    public:
#define SYRAH_NEON_LOOP(index) SYRAH_UNROLL(N/4) \
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

    // in _casts.h
    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<float, N, true>& v);

    SYRAH_FORCEINLINE FixedVector(const FixedVector<int, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_NEON_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    static SYRAH_FORCEINLINE FixedVector<int, N, true> reinterpret(const FixedVector<float, N, true>& v);

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    static SYRAH_FORCEINLINE FixedVector<int, N, true> Zero() {
      FixedVector<int, N, true> result;
      int32x4_t zero_int;
      zero_int = veorq_s32(zero_int, zero_int);
      SYRAH_NEON_LOOP(i) {
        result.data[i] = zero_int;
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
      int32x4_t splat = vdupq_n_s32(value);
      SYRAH_NEON_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const int* values) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vld1q_s32(&values[4*i]);
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const int* values) {
      load(values);
    }

    SYRAH_FORCEINLINE void load(const int* values, const FixedVectorMask<N>& mask,
                               const int default_value) {
      int32x4_t default_data = vdupq_n_s32(default_value);
      SYRAH_NEON_LOOP(i) {
        int32x4_t loaded_data = vld1q_s32(&(values[4*i]));
        data[i] = vbslq_s32(mask.data[i], loaded_data, default_data);
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const int* values, const FixedVectorMask<N>& mask,
                                       const int default_value) {
      load(values, mask, default_value);
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
      SYRAH_NEON_LOOP(i) {
        vst1q_s32(&(dst[4*i]), data[i]);
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst) const {
      store(dst);
    }

    SYRAH_FORCEINLINE void store(int* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_NEON_LOOP(i) {
        // Consider maskmov
        int32x4_t cur_val = vld1q_s32(&(dst[4*i]));
        vst1q_s32(&(dst[4*i]), vbslq_s32(mask.data[i], data[i], cur_val));
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst, const FixedVectorMask<N>& mask) const {
      store(dst, mask);
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
      SYRAH_NEON_LOOP(i) {
        data[i] = vbslq_s32(mask.data[i], v.data[i], data[i]);
      }
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator+(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vaddq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator+=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vaddq_s32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator-() const {
      FixedVector<int, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vnegq_s32(data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator-(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vsubq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator-=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vsubq_s32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator*(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vmulq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator*=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vmulq_s32(data[i], v.data[i]);
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
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vandq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator&(const int& scalar) const {
      FixedVector<int, N, true> result;
      const int32x4_t splat = vdupq_n_s32(scalar);
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vandq_s32(data[i], splat);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator&=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vandq_s32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator&=(const int& scalar) {
      const int32x4_t splat = vdupq_n_s32(scalar);
      SYRAH_NEON_LOOP(i) {
        data[i] = vandq_s32(data[i], splat);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator<<(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vshlq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator<<=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vshlq_s32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator<<(int v) const {
      FixedVector<int, N, true> result;
      int32x4_t shift_amt = vdupq_n_s32(v);
      SYRAH_NEON_LOOP(i) {
        // NOTE(boulos): If v was a known immediate you could use
        // vshlq_n_s32... vec.shl<imm>() would work but is unlikely to
        // be used.
        result.data[i] = vshlq_s32(data[i], shift_amt);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator<<=(int v) {
      int32x4_t shift_amt = vdupq_n_s32(v);
      SYRAH_NEON_LOOP(i) {
        // NOTE(boulos): If v was a known immediate you could use
        // vshlq_n_s32... vec.shl<imm>() would work but is unlikely to
        // be used.
        data[i] = vshlq_s32(data[i], shift_amt);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator>>(int v) const {
      FixedVector<int, N, true> result;
      // Shift right is shift by negative amount
      int32x4_t shift_amt = vdupq_n_s32(-v);
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vshlq_s32(data[i], shift_amt);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator>>=(int v) {
      // Shift right is shift by negative amount
      int32x4_t shift_amt = vdupq_n_s32(-v);
      SYRAH_NEON_LOOP(i) {
        data[i] = vshlq_s32(data[i], shift_amt);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator>>(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vshlq_s32(data[i], vnegq_s32(v.data[i]));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator>>=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vshlq_s32(data[i], vnegq_s32(v.data[i]));
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator|(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vorrq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator|=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vorrq_s32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true> operator^(const FixedVector<int, N, true>& v) const {
      FixedVector<int, N> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = veorq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<int, N, true>& operator^=(const FixedVector<int, N, true>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = veorq_s32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcltq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcleq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vceqq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<int, N, true>& v) const {
      return !operator==(v);
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcgeq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<int, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcgtq_s32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE int MaxElement() const {
      int32x4_t tmpMax = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMax = vmaxq_s32(data[i], tmpMax);
      }
      int32x2_t lo_part = vget_low_s32(tmpMax);
      int32x2_t hi_part = vget_high_s32(tmpMax);
      int32x2_t fold_max = vpmax_s32(lo_part, hi_part);
      fold_max = vpmax_s32(fold_max, fold_max);
      return vget_lane_s32(fold_max, 0);
    }

    SYRAH_FORCEINLINE int MinElement() const {
      int32x4_t tmpMin = data[0];
      for (int i = 1; i < N/4; i++) {
        tmpMin = vminq_s32(data[i], tmpMin);
      }
      int32x2_t lo_part = vget_low_s32(tmpMin);
      int32x2_t hi_part = vget_high_s32(tmpMin);
      int32x2_t fold_min = vpmin_s32(lo_part, hi_part);
      fold_min = vpmin_s32(fold_min, fold_min);
      return vget_lane_s32(fold_min, 0);
    }

    SYRAH_FORCEINLINE int foldMin() const { return MinElement(); }
    SYRAH_FORCEINLINE int foldMax() const { return MaxElement(); }

    SYRAH_FORCEINLINE int foldSum() const {
      int32x4_t tmp_sum;
      tmp_sum = veorq_s32(tmp_sum, tmp_sum);
      SYRAH_NEON_LOOP(i) {
        tmp_sum = vaddq_s32(tmp_sum, data[i]);
      }

      // [a, b]
      int32x2_t lo_part = vget_low_s32(tmp_sum);
      // [c, d]
      int32x2_t hi_part = vget_high_s32(tmp_sum);
      // [a+c, b+d]
      int32x2_t fold_sum = vpadd_s32(lo_part, hi_part);
      // [(a+c)+(b+d), (a+c)+(b+d)]
      fold_sum = vpadd_s32(fold_sum, fold_sum);
      return vget_lane_s32(fold_sum, 0);
    }

    SYRAH_FORCEINLINE int foldProd() const {
      int32x4_t tmp_prod = vdupq_n_s32(1);
      SYRAH_NEON_LOOP(i) {
        tmp_prod = vmulq_s32(tmp_prod, data[i]);
      }

      // [a, b, c, d]
      int32x2_t ab = vget_low_s32(tmp_prod);
      int32x2_t cd = vget_high_s32(tmp_prod);
      int32x2_t ac_bd = vmul_s32(ab, cd);
      int32x2_t bd = vdup_lane_s32(ac_bd, 1);
      int32x2_t acbd = vmul_s32(ac_bd, bd);
      return vget_lane_s32(acbd, 0);
    }

    int32x4_t data[N/4];
  };

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator+(const int& s, const FixedVector<int, N, true>& v) {
    FixedVector<int, N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vaddq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator+(const FixedVector<int, N, true>& v, const int& s) {
    FixedVector<int, N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vaddq_s32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator-(const int& s, const FixedVector<int, N, true>& v) {
    FixedVector<int, N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vsubq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator-(const FixedVector<int, N, true>& v, const int& s) {
    FixedVector<int, N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vsubq_s32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator*(const int& s, const FixedVector<int, N, true>& v) {
    FixedVector<int, N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmulq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> operator*(const FixedVector<int, N, true>& v, const int& s) {
    FixedVector<int, N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmulq_s32(v.data[i], splat);
    }
    return result;
  }

  // NOTE(boulos): Explicitly avoiding overriding the operator/
  // here. It'll get handled by the scalar codepath which is what we
  // want anyway.

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcltq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcltq_s32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcleq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator<=(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcleq_s32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vceqq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator==(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vceqq_s32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgeq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>=(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgeq_s32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const int& s, const FixedVector<int, N, true>& v) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgtq_s32(splat, v.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator>(const FixedVector<int, N, true>& v, const int& s) {
    FixedVectorMask<N> result;
    const int32x4_t splat = vdupq_n_s32(s);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vcgtq_s32(v.data[i], splat);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const int& s, const FixedVector<int, N, true>& v) {
    return !(operator==(s,v));
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<int, N, true>& v, const int& s) {
    return !(operator==(v, s));
  }


  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> max(const FixedVector<int, N, true>& v1, const FixedVector<int, N, true>& v2) {
    FixedVector<int, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmaxq_s32(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> min(const FixedVector<int, N, true>& v1, const FixedVector<int, N, true>& v2) {
    FixedVector<int, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vminq_s32(v1.data[i], v2.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> madd(const FixedVector<int, N, true>& a,
                                                   const FixedVector<int, N, true>& b,
                                                   const FixedVector<int, N, true>& c) {
    FixedVector<int, N, true> result;
    SYRAH_NEON_LOOP(i) {
      //result.data[i] = vmlaq_s32(a.data[i], b.data[i], c.data[i]);
      //result.data[i] = vmlaq_s32(b.data[i], c.data[i], a.data[i]);
      result.data[i] = vaddq_s32(c.data[i], vmulq_s32(a.data[i], b.data[i]));
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
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vbslq_s32(mask.data[i], a.data[i], b.data[i]);
    }
    return result;
  }



#undef SYRAH_NEON_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_NEON_INT_H_
