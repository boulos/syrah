#ifndef _SYRAH_FIXED_VECTOR_NEON_CASTS_H_
#define _SYRAH_FIXED_VECTOR_NEON_CASTS_H_

namespace syrah {
#define SYRAH_NEON_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)

  // float to int
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true>::FixedVector(const FixedVector<float, N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vcvtq_s32_f32(v.data[i]);
    }
  }

  // int to uint8
  template<int N>
  SYRAH_FORCEINLINE FixedVector<uint8_t, N, true>::FixedVector(const FixedVector<int, N, true>& v) {
     for (int i = 0; i < N/16; i++) {
        uint16x4_t a_16 = vqmovun_s32(v.data[4*i + 0]);
        uint16x4_t b_16 = vqmovun_s32(v.data[4*i + 1]);
        uint16x4_t c_16 = vqmovun_s32(v.data[4*i + 2]);
        uint16x4_t d_16 = vqmovun_s32(v.data[4*i + 3]);

        uint16x8_t ab_16 = vcombine_u16(a_16, b_16);
        uint16x8_t cd_16 = vcombine_u16(c_16, d_16);

        uint8x8_t ab_8 = vqmovn_u16(ab_16);
        uint8x8_t cd_8 = vqmovn_u16(cd_16);

        data[i] = vcombine_u8(ab_8, cd_8);
     }
  }


  // int = reinterpret_cast<int>(float)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> FixedVector<int, N, true>::reinterpret(const FixedVector<float, N, true>& v) {
    FixedVector<int, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vreinterpretq_s32_f32(v.data[i]);
    }
    return result;
  }

  // int to float
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true>::FixedVector(const FixedVector<int, N> &v) {
    SYRAH_NEON_LOOP(i) {
      data[i] = vcvtq_f32_s32(v.data[i]);
    }
  }

  // float = reinterpret_cast<float>(int)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> FixedVector<float, N, true>::reinterpret(const FixedVector<int, N, true>& v) {
    FixedVector<float, N, true> result;
    SYRAH_NEON_LOOP(i) {
      result.data[i] = vreinterpretq_f32_s32(v.data[i]);
    }
    return result;
  }
}

#undef SYRAH_NEON_LOOP

#endif // _SYRAH_FIXED_VECTOR_NEON_CASTS_H_
