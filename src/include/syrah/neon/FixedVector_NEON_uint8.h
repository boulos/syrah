#ifndef _SYRAH_FIXED_VECTOR_NEON_UINT8_H_
#define _SYRAH_FIXED_VECTOR_NEON_UINT8_H_

namespace syrah {

#define SYRAH_COLLAPSE_CHAR(mask, which) vdupq_n_u32((uint32_t)vgetq_lane_u8(mask, which))

  // We want the first 8-bits from each element of each vector
  // lane...
  SYRAH_FORCEINLINE uint8x16_t ExpandChars(uint32x4_t a,
                                          uint32x4_t b,
                                          uint32x4_t c,
                                          uint32x4_t d) {
    // Get first halfs of all the elements
    uint16x4_t a_16 = vqmovn_u32(a);
    uint16x4_t b_16 = vqmovn_u32(b);
    uint16x4_t c_16 = vqmovn_u32(c);
    uint16x4_t d_16 = vqmovn_u32(d);

    // Combine the 2 halves (of halves)
    uint16x8_t ab_16 = vcombine_u16(a_16, b_16);
    uint16x8_t cd_16 = vcombine_u16(c_16, d_16);

    // Get the first halfs of those bits
    uint8x8_t ab_8 = vmovn_u16(ab_16);
    uint8x8_t cd_8 = vmovn_u16(cd_16);

    // Combine the results
    return vcombine_u8(ab_8, cd_8);
  }

  template<int N>
  class SYRAH_ALIGN(16) FixedVector<uint8_t, N, true> {
  public:
#define SYRAH_NEON_LOOP(index) SYRAH_UNROLL(N/16)              \
      for (int index = 0; index < N/16; index++)

  SYRAH_FORCEINLINE FixedVector() {}

  SYRAH_FORCEINLINE FixedVector(const uint8_t value) {
    load(value);
  }

  SYRAH_FORCEINLINE FixedVector(const uint8_t* values) {
    load(values);
  }

  SYRAH_FORCEINLINE FixedVector(const uint8_t* values, bool aligned) {
    load_aligned(values);
  }

  SYRAH_FORCEINLINE FixedVector(const uint8_t* values, const FixedVectorMask<N>& mask,
                               const uint8_t default_value) {
    load(values, mask, default_value);
  }

  SYRAH_FORCEINLINE FixedVector(const uint8_t* values, const FixedVectorMask<N>& mask,
                               const uint8_t default_value, bool aligned) {
    load_aligned(values, mask, default_value);
  }

  // in _casts.h
  SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<float, N, true>& v);

  // in _casts.h
  SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<int, N, true>& v);

  SYRAH_FORCEINLINE FixedVector(const FixedVector<uint8_t, N, true>& v) {
    // TODO(boulos): Optimize this
    SYRAH_NEON_LOOP(i) {
      data[i] = v.data[i];
    }
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = v.data[i];
    }
    return *this;
  }

  static SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> Zero() {
    FixedVector<uint8_t, N, true> result;
    uint8x16_t zero_int;
    zero_int = veorq_u8(zero_int, zero_int);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = zero_int;
    }
    return result;
  }


  SYRAH_FORCEINLINE FixedVector(const uint8_t* base, const FixedVector<int, N, true>& offsets, const int scale) {
    gather(base, offsets, scale);
  }

  SYRAH_FORCEINLINE FixedVector(const uint8_t* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
    gather(base, offsets, scale, mask);
  }

  SYRAH_FORCEINLINE FixedVector(const char* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
    gather(base, offsets, scale, mask);
  }

  SYRAH_FORCEINLINE const uint8_t& operator[](uint8_t i) const {
    return ((uint8_t*)data)[i];
  }

  SYRAH_FORCEINLINE uint8_t& operator[](uint8_t i) {
    return ((uint8_t*)data)[i];
  }

  SYRAH_FORCEINLINE void load(const uint8_t value) {
    uint8x16_t splat = vmovq_n_u8(value);
    SYRAH_NEON_LOOP(i) {
      data[i] = splat;
    }
  }

  SYRAH_FORCEINLINE void load(const uint8_t* values) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vld1q_u8(&values[16*i]);
    }
  }

  SYRAH_FORCEINLINE void load_aligned(const uint8_t* values) {
    load(values);
  }

  SYRAH_FORCEINLINE void load(const uint8_t* values, const FixedVectorMask<N>& mask,
                             const uint8_t default_value) {
    uint8x16_t default_data = vmovq_n_u8(default_value);
    SYRAH_NEON_LOOP(i) {
      uint8x16_t loaded_data = vld1q_u8(&(values[16*i]));
      uint8x16_t mask_char = ExpandChars(mask.data[(i/4) + 0],
                                         mask.data[(i/4) + 1],
                                         mask.data[(i/4) + 2],
                                         mask.data[(i/4) + 3]);
      data[i] = vbslq_u8(mask_char, loaded_data, default_data);
    }
  }

  SYRAH_FORCEINLINE void load_aligned(const uint8_t* values, const FixedVectorMask<N>& mask,
                                     const uint8_t default_value) {
    load(values, mask, default_value);
  }

  template <typename T>
    SYRAH_FORCEINLINE void gather(const T* base, const FixedVector<int, N, true>& offsets, const int scale) {
    uint8_t* int_data = reinterpret_cast<uint8_t*>(data);
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
    uint8_t* int_data = reinterpret_cast<uint8_t*>(data);
    for (int i = 0; i < N; i++) {
      if (mask.get(i)) {
        const T* addr = reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + offsets[i] * scale);
        int_data[i] = static_cast<int>(*addr);
      }
    }
  }

  SYRAH_FORCEINLINE void store(uint8_t* dst) const {
    SYRAH_NEON_LOOP(i) {
      vst1q_u8(&(dst[16*i]), data[i]);
    }
  }

  SYRAH_FORCEINLINE void store_aligned(uint8_t* dst) const {
    store(dst);
  }

  SYRAH_FORCEINLINE void store(uint8_t* dst, const FixedVectorMask<N>& mask) const {
    SYRAH_NEON_LOOP(i) {
      // Consider maskmov
      uint8x16_t cur_val = vld1q_u8(&(dst[16*i]));
      uint8x16_t mask_char = ExpandChars(mask.data[(i/4) + 0],
                                         mask.data[(i/4) + 1],
                                         mask.data[(i/4) + 2],
                                         mask.data[(i/4) + 3]);
      vst1q_u8(&(dst[16*i]), vbslq_u8(mask_char, data[i], cur_val));
    }
  }

  SYRAH_FORCEINLINE void store_aligned(uint8_t* dst, const FixedVectorMask<N>& mask) const {
    store(dst, mask);
  }

  SYRAH_FORCEINLINE void scatter(uint8_t* base, const FixedVector<int, N, true>& offsets, const int scale) const {
    const uint8_t* int_data = reinterpret_cast<const uint8_t*>(data);
    // TODO(boulos): Consider doing all the indexing ops with
    // vector? Seems pointless.
    for (int i = 0; i < N; i++) {
      uint8_t* addr = reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
      *addr = int_data[i];
    }
  }

  // QUESTION(boulos): Constant scale only?
  SYRAH_FORCEINLINE void scatter(uint8_t* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) const {
    const uint8_t* int_data = reinterpret_cast<const uint8_t*>(data);
    for (int i = 0; i < N; i++) {
      if (mask.get(i)) {
        uint8_t* addr = reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(base) + offsets[i] * scale);
        *addr = int_data[i];
      }
    }
  }


  SYRAH_FORCEINLINE void merge(const FixedVector<uint8_t, N, true>& v,
                              const FixedVectorMask<N, true>& mask) {
    SYRAH_NEON_LOOP(i) {
      uint8x16_t mask_char = ExpandChars(mask.data[(i/4) + 0],
                                         mask.data[(i/4) + 1],
                                         mask.data[(i/4) + 2],
                                         mask.data[(i/4) + 3]);
      data[i] = vbslq_u8(mask_char, v.data[i], data[i]);
    }
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator+(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vaddq_u8(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator+=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vaddq_u8(data[i], v.data[i]);
    }
    return *this;
  }

#if 0
  // TODO(boulos): Decide what negation means for unsigned?
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator-() const {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vnegq_u8(data[i]);
    }
    return result;
  }
#endif

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator-(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vsubq_u8(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator-=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vsubq_u8(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator*(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmulq_u8(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator*=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vmulq_u8(data[i], v.data[i]);
    }
    return *this;
  }


  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator/(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    const uint8_t* int_data = reinterpret_cast<const uint8_t*>(data);
    for (int i = 0; i < N; i++) {
      result[i] = int_data[i] / v[i];
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator/=(const FixedVector<uint8_t, N, true>& v) {
    uint8_t* int_data = reinterpret_cast<uint8_t*>(data);
    for (int i = 0; i < N; i++) {
      int_data[i] /= v[i];
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator%(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    const uint8_t* int_data = reinterpret_cast<const uint8_t*>(data);
    for (int i = 0; i < N; i++) {
      result[i] = int_data[i] % v[i];
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator%=(const FixedVector<uint8_t, N, true>& v) {
    uint8_t* int_data = reinterpret_cast<uint8_t*>(data);
    for (int i = 0; i < N; i++) {
      int_data[i] %= v[i];
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator&(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vandq_u8(data[i], v.data[i]);
    }
    return result;
  }

  // QUESTION(boulos): Will adding an explicit "constant int"
  // version allow me to improve this?
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator&=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vandq_u8(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator<<(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vshlq_u8(data[i], vreinterpretq_s8_u8(v.data[i]));
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator<<=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vshlq_u8(data[i], vreinterpretq_s8_u8(v.data[i]));
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator<<(int8_t v) const {
    FixedVector<uint8_t, N, true> result;
    int8x16_t shift_amt = vmovq_n_s8(v);
    SYRAH_NEON_LOOP(i) {
      // NOTE(boulos): If v was a known immediate you could use
      // vshlq_n_u8... vec.shl<imm>() would work but is unlikely to
      // be used.
      result.data[i] = vshlq_u8(data[i], shift_amt);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator<<=(int8_t v) {
    int8x16_t shift_amt = vmovq_n_s8(v);
    SYRAH_NEON_LOOP(i) {
      // NOTE(boulos): If v was a known immediate you could use
      // vshlq_n_u8... vec.shl<imm>() would work but is unlikely to
      // be used.
      data[i] = vshlq_u8(data[i], shift_amt);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator>>(int8_t v) const {
    FixedVector<uint8_t, N, true> result;
    // Shift right is shift by negative amount
    int8x16_t shift_amt = vmovq_n_s8(-v);
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vshlq_u8(data[i], shift_amt);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator>>=(int8_t v) {
    // Shift right is shift by negative amount
    int8x16_t shift_amt = vmovq_n_s8(-v);
    SYRAH_NEON_LOOP(i) {
      data[i] = vshlq_u8(data[i], shift_amt);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator>>(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vshlq_u8(data[i], vnegq_s8(vreinterpretq_s8_u8(v.data[i])));
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator>>=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vshlq_u8(data[i], vnegq_s8(vreinterpretq_s8_u8(v.data[i])));
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator|(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vorrq_u8(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator|=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vorrq_u8(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> operator^(const FixedVector<uint8_t, N, true>& v) const {
    FixedVector<uint8_t, N> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = veorq_u8(data[i], v.data[i]);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>& operator^=(const FixedVector<uint8_t, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = veorq_u8(data[i], v.data[i]);
    }
    return *this;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<uint8_t, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_NEON_LOOP(i) {
      uint8x16_t mask_char = vcltq_u8(data[i], v.data[i]);
      result.data[i/4 + 0] = SYRAH_COLLAPSE_CHAR(mask_char, 0);
      result.data[i/4 + 1] = SYRAH_COLLAPSE_CHAR(mask_char, 1);
      result.data[i/4 + 2] = SYRAH_COLLAPSE_CHAR(mask_char, 2);
      result.data[i/4 + 3] = SYRAH_COLLAPSE_CHAR(mask_char, 3);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<uint8_t, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_NEON_LOOP(i) {
      uint8x16_t mask_char = vcleq_u8(data[i], v.data[i]);
      result.data[i/4 + 0] = SYRAH_COLLAPSE_CHAR(mask_char, 0);
      result.data[i/4 + 1] = SYRAH_COLLAPSE_CHAR(mask_char, 1);
      result.data[i/4 + 2] = SYRAH_COLLAPSE_CHAR(mask_char, 2);
      result.data[i/4 + 3] = SYRAH_COLLAPSE_CHAR(mask_char, 3);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<uint8_t, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_NEON_LOOP(i) {
      uint8x16_t mask_char = vceqq_u8(data[i], v.data[i]);
      result.data[i/4 + 0] = SYRAH_COLLAPSE_CHAR(mask_char, 0);
      result.data[i/4 + 1] = SYRAH_COLLAPSE_CHAR(mask_char, 1);
      result.data[i/4 + 2] = SYRAH_COLLAPSE_CHAR(mask_char, 2);
      result.data[i/4 + 3] = SYRAH_COLLAPSE_CHAR(mask_char, 3);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator !=(const FixedVector<uint8_t, N, true>& v) const {
    return !operator==(v);
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<uint8_t, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_NEON_LOOP(i) {
      uint8x16_t mask_char = vcgeq_u8(data[i], v.data[i]);
      result.data[i/4 + 0] = SYRAH_COLLAPSE_CHAR(mask_char, 0);
      result.data[i/4 + 1] = SYRAH_COLLAPSE_CHAR(mask_char, 1);
      result.data[i/4 + 2] = SYRAH_COLLAPSE_CHAR(mask_char, 2);
      result.data[i/4 + 3] = SYRAH_COLLAPSE_CHAR(mask_char, 3);
    }
    return result;
  }

  SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<uint8_t, N, true>& v) const {
    FixedVectorMask<N, true> result;
    SYRAH_NEON_LOOP(i) {
      uint8x16_t mask_char = vcgtq_u8(data[i], v.data[i]);
      result.data[i/4 + 0] = SYRAH_COLLAPSE_CHAR(mask_char, 0);
      result.data[i/4 + 1] = SYRAH_COLLAPSE_CHAR(mask_char, 1);
      result.data[i/4 + 2] = SYRAH_COLLAPSE_CHAR(mask_char, 2);
      result.data[i/4 + 3] = SYRAH_COLLAPSE_CHAR(mask_char, 3);
    }
    return result;
  }

  SYRAH_FORCEINLINE uint8_t MaxElement() const {
    uint8x16_t tmpMax = data[0];
    for (int i = 1; i < N/16; i++) {
      tmpMax = vmaxq_u8(data[i], tmpMax);
    }
    // get first 8 and second 8
    uint8x8_t lo_part = vget_low_u8(tmpMax);
    uint8x8_t hi_part = vget_high_u8(tmpMax);
    // get max of first8/second8
    uint8x8_t fold_max = vpmax_u8(lo_part, hi_part);
    // folding max of 4/4
    fold_max = vpmax_u8(fold_max, fold_max);
    // folding max of 2/2
    fold_max = vpmax_u8(fold_max, fold_max);
    // folding max of 1
    fold_max = vpmax_u8(fold_max, fold_max);
    return vget_lane_u8(fold_max, 0);
  }

  SYRAH_FORCEINLINE uint8_t MinElement() const {
    uint8x16_t tmpMin = data[0];
    for (int i = 1; i < N/16; i++) {
      tmpMin = vminq_u8(data[i], tmpMin);
    }
    uint8x8_t lo_part = vget_low_u8(tmpMin);
    uint8x8_t hi_part = vget_high_u8(tmpMin);
    uint8x8_t fold_min = vpmin_u8(lo_part, hi_part);
    fold_min = vpmin_u8(fold_min, fold_min);
    fold_min = vpmin_u8(fold_min, fold_min);
    fold_min = vpmin_u8(fold_min, fold_min);
    return vget_lane_u8(fold_min, 0);
  }

  SYRAH_FORCEINLINE uint8_t foldMin() const { return MinElement(); }
  SYRAH_FORCEINLINE uint8_t foldMax() const { return MaxElement(); }

  SYRAH_FORCEINLINE uint8_t foldSum() const {
    uint8x16_t tmp_sum;
    tmp_sum = veorq_u8(tmp_sum, tmp_sum);
    SYRAH_NEON_LOOP(i) {
      tmp_sum = vaddq_u8(tmp_sum, data[i]);
    }

    // [a, b, c, d, e, f, g, h]
    uint8x8_t lo_part = vget_low_u8(tmp_sum);
    // [i, j, k, l, m, n, o, p]
    uint8x8_t hi_part = vget_high_u8(tmp_sum);
    // pairs. [a+i, b+j, c+k, ... h+p]
    uint8x8_t fold_sum = vpadd_u8(lo_part, hi_part);
    // "quads". [a+i+b+j, ...]
    fold_sum = vpadd_u8(fold_sum, fold_sum);
    // "eights"
    fold_sum = vpadd_u8(fold_sum, fold_sum);
    // all 16
    fold_sum = vpadd_u8(fold_sum, fold_sum);
    return vget_lane_u8(fold_sum, 0);
  }

  SYRAH_FORCEINLINE uint8_t foldProd() const {
    uint8x16_t tmp_prod = vdupq_n_u8(1);
    SYRAH_NEON_LOOP(i) {
      tmp_prod = vmulq_u8(tmp_prod, data[i]);
    }

    // [a, b, c, d, e, f, g, h]
    uint8x8_t lo_part = vget_low_u8(tmp_prod);
    // [i, j, k, l, m, n, o, p]
    uint8x8_t hi_part = vget_high_u8(tmp_prod);
    // pairs. [a*i, b*j, c*k, d*l, e*m, f*n, g*o, h*p]
    uint8x8_t fold_mul = vmul_u8(lo_part, hi_part);
    // Get the second half into the front
    // [e*m, f*n, g*o, h*p, a*i, b*j, c*k, d*l]
    uint8x8_t upper_half = vext_u8(fold_mul, fold_mul, 4);
    // [aiem, bjfn, ckgo, dlhp, emai, fnbj, gock, hpdl]
    fold_mul = vmul_u8(fold_mul, upper_half);
    // Now just get the top two out (gock and hpdl):
    // [gock, hpdl, aiem, bjfn, ckgo, dlhp, emai, fnjb]
    uint8x8_t upper_two = vext_u8(fold_mul, fold_mul, 2);
    // [aiemgock, bjfnhpdl, ....]
    fold_mul = vmul_u8(fold_mul, upper_two);
    // grab out bjfnhpdl and multiply
    fold_mul = vmul_u8(fold_mul, vdup_lane_u8(fold_mul, 1));
    return vget_lane_u8(fold_mul, 0);
  }

  uint8x16_t data[N/16];
};

  template<int N>
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> max(const FixedVector<uint8_t, N, true>& v1, const FixedVector<uint8_t, N, true>& v2) {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vmaxq_u8(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> min(const FixedVector<uint8_t, N, true>& v1, const FixedVector<uint8_t, N, true>& v2) {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vminq_u8(v1.data[i], v2.data[i]);
    }
    return result;
  }

  // TODO(boulos): Same issue here. Handle scalar BINOP vector and vector BINOP scalar.

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> madd(const FixedVector<uint8_t, N, true>& a,
                                                      const FixedVector<uint8_t, N, true>& b,
                                                      const FixedVector<uint8_t, N, true>& c) {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vaddq_u8(c.data[i], vmulq_u8(a.data[i], b.data[i]));
    }
    return result;
  }

  // truncate(a) = a
  template<int N>
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> trunc(const FixedVector<uint8_t, N, true>& a) {
    return a;
  }

  // rint(a) = a
  template<int N>
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> rint(const FixedVector<uint8_t, N, true>& a) {
    return a;
  }

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true> select(const FixedVector<uint8_t, N, true>& a,
                                                        const FixedVector<uint8_t, N, true>& b,
                                                        const FixedVectorMask<N>& mask) {
    FixedVector<uint8_t, N, true> result;
    SYRAH_NEON_LOOP(i) {
      uint8x16_t mask_char = ExpandChars(mask.data[(i/4) + 0],
                                         mask.data[(i/4) + 1],
                                         mask.data[(i/4) + 2],
                                         mask.data[(i/4) + 3]);
      result.data[i] = vbslq_u8(mask_char, a.data[i], b.data[i]);
    }
    return result;
  }



#undef SYRAH_NEON_LOOP
#undef SYRAH_COLLAPSE_CHAR
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_NEON_INT_H_
