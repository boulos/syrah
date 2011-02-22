#ifndef _SYRAH_FIXED_VECTOR_NEON_MASK_H_
#define _SYRAH_FIXED_VECTOR_NEON_MASK_H_

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(16) FixedVectorMask<N, true> {
  public:
#define SYRAH_NEON_LOOP(index) SYRAH_UNROLL(N/4) \
for (int index = 0; index < N/4; index++)

    SYRAH_FORCEINLINE FixedVectorMask(bool value) {
      uint32x4_t zero_int;
      zero_int = veorq_u32(zero_int, zero_int);
      if (true) {
        uint32x4_t all_on = vceqq_u32(zero_int, zero_int);
        SYRAH_NEON_LOOP(i) {
          data[i] = all_on;
        }
      } else {
        SYRAH_NEON_LOOP(i) {
          data[i] = zero_int;
        }
      }
    }
    SYRAH_FORCEINLINE FixedVectorMask() {}
    SYRAH_FORCEINLINE FixedVectorMask(const bool* values) {
      const unsigned int kTrue = static_cast<unsigned int>(-1);
      unsigned int* int_data = reinterpret_cast<unsigned int*>(data);
      for (int i = 0; i < N; i++) {
        int_data[i] = (values[i]) ? kTrue : 0;
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask(const unsigned int* values) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vld1q_u32(&values[4*i]);
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask(const FixedVectorMask<N>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator=(const FixedVectorMask<N>& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    // NOTE(boulos): Could also try to do fancy stuff like STL bit
    // vector does to allow operator[].
    SYRAH_FORCEINLINE bool get(int i) const {
      const unsigned int kTrue = static_cast<unsigned int>(-1);
      switch (i & 3) {
      case 0:  return kTrue == vgetq_lane_u32(data[i/4], 0);
      case 1:  return kTrue == vgetq_lane_u32(data[i/4], 1);
      case 2:  return kTrue == vgetq_lane_u32(data[i/4], 2);
      default: return kTrue == vgetq_lane_u32(data[i/4], 3);
      }
      return false;
    }

    SYRAH_FORCEINLINE void set(int i, bool val) {
      const unsigned int kTrue = static_cast<unsigned int>(-1);
      unsigned int int_val = (val) ? kTrue : 0;
      switch (i & 3) {
      case 0:  data[i/4] = vsetq_lane_u32(int_val, data[i/4], 0);
      case 1:  data[i/4] = vsetq_lane_u32(int_val, data[i/4], 1);
      case 2:  data[i/4] = vsetq_lane_u32(int_val, data[i/4], 2);
      default: data[i/4] = vsetq_lane_u32(int_val, data[i/4], 3);
      }
    }

    // NOTE(boulos): I'm intentionally not overloading || as doing so
    // doesn't result in short-circuiting for overloaded ||:
    // http://www.parashift.com/c++-faq-lite/operator-overloading.html#faq-13.9
    // part 19.
    SYRAH_FORCEINLINE FixedVectorMask operator|(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vorrq_u32(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator|=(const FixedVectorMask& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vorrq_u32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator&(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vandq_u32(data[i],v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator&=(const FixedVectorMask& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = vandq_u32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator^(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = veorq_u32(data[i],v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator^=(const FixedVectorMask& v) {
      SYRAH_NEON_LOOP(i) {
        data[i] = veorq_u32(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE static FixedVectorMask True() {
      // NOTE(boulos): Don't just call return FixedVectorMask(true);
      // to avoid branch.
      FixedVectorMask result;
      uint32x4_t zero_int;
      zero_int = veorq_u32(zero_int, zero_int);
      uint32x4_t all_on = vceqq_u32(zero_int, zero_int);
      SYRAH_NEON_LOOP(i) {
        result.data[i] = all_on;
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask False() {
      FixedVectorMask result;
      uint32x4_t zero_int;
      zero_int = veorq_u32(zero_int, zero_int);
      SYRAH_NEON_LOOP(i) {
        result.data[i] = zero_int;
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask FirstN(int n) {
      uint32x4_t n_splat = vmovq_n_u32(static_cast<unsigned int>(n));
      const SYRAH_ALIGN(16) unsigned int seq_start[] = {0, 1, 2, 3};
      uint32x4_t small_seq = vld1q_u32(seq_start);
      uint32x4_t inc_val = vmovq_n_u32(4);
      FixedVectorMask result;
      SYRAH_NEON_LOOP(i) {
        result.data[i] = vcltq_u32(small_seq, n_splat);
        small_seq = vaddq_u32(small_seq, inc_val);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator!() const {
      FixedVectorMask result;
      SYRAH_NEON_LOOP(i) {
        // NOTE(boulos): NEON's andnot function is actually (NOT x) & y
        result.data[i] = vmvnq_u32(data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst) const {
      SYRAH_NEON_LOOP(i) {
        vst1q_f32(&(dst[4*i]), vreinterpretq_f32_u32(data[i]));
      }
    }

    SYRAH_FORCEINLINE void store_aligned(int* dst) const {
      SYRAH_NEON_LOOP(i) {
        vst1q_u32(&(dst[4*i]), data[i]);
      }
    }

    uint32x4_t data[N/4];
  };

  template<int N>
  SYRAH_FORCEINLINE bool All(const FixedVectorMask<N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      if (vgetq_lane_s32(vclsq_s32(vreinterpretq_s32_u32(v.data[i])), 3) != 4) return false;
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE bool Any(const FixedVectorMask<N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      if (vgetq_lane_s32(vclsq_s32(vreinterpretq_s32_u32(v.data[i])), 3) != 0) return true;
    }
    return false;
  }

  template<int N>
  SYRAH_FORCEINLINE bool None(const FixedVectorMask<N, true>& v) {
    SYRAH_NEON_LOOP(i) {
      if (vgetq_lane_s32(vclsq_s32(vreinterpretq_s32_u32(v.data[i])), 3) != 0) return false;
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE int NumActive(const FixedVectorMask<N, true>& v) {
    int32x4_t sign_count;
    sign_count = veorq_s32(sign_count, sign_count);
    SYRAH_NEON_LOOP(i) {
      sign_count = vaddq_s32(sign_count, vclsq_s32(vreinterpretq_s32_u32(v.data[i])));
    }
    // I'm guessing that sign_count[3] is the total sum?
    return vgetq_lane_s32(sign_count, 3);
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<int, N, true> PrefixSum(const FixedVectorMask<N, true>& v) {
    FixedVector<int, N, true> result;
    int32x4_t sign_count;
    sign_count = veorq_s32(sign_count, sign_count);
    for (int i = 0; i < N; i++) {
      sign_count = vaddq_s32(sign_count, vclsq_s32(vreinterpretq_s32_u32(v.data[i])));
      result[i] = sign_count;
    }
    return result;
  }

#undef SYRAH_NEON_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_NEON_MASK_H_
