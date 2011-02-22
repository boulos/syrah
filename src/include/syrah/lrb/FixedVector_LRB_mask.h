#ifndef _SYRAH_FIXED_VECTOR_LRB_MASK_H_
#define _SYRAH_FIXED_VECTOR_LRB_MASK_H_

namespace syrah {
  template <int N>
  class SYRAH_ALIGN(64) FixedVectorMask<N, true> {
  public:
#define SYRAH_LRB_LOOP(index) SYRAH_UNROLL(N/16) \
for (int index = 0; index < N/16; index++)

    SYRAH_FORCEINLINE FixedVectorMask(bool value) {
      __mmask bitvec = _mm512_int2mask((value) ? ~0x0 : 0);
      SYRAH_LRB_LOOP(i) {
        data[i] = bitvec;
      }
    }

    SYRAH_FORCEINLINE FixedVectorMask() {}
    SYRAH_FORCEINLINE FixedVectorMask(const bool* values) {
      SYRAH_LRB_LOOP(i) {
        __mmask bitvec = 0;
        for (int j = 0; j < 16; j++) {
          bitvec |= ((values[16*i + j]) ? (1 << j) : 0);
        }
        data[i] = bitvec;
      }
    }
    SYRAH_FORCEINLINE FixedVectorMask(const FixedVectorMask<N>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = v.data[i];
      }
    }
    SYRAH_FORCEINLINE FixedVectorMask& operator=(const FixedVectorMask<N>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    SYRAH_FORCEINLINE bool get(int i) const {
      int which = i >> 4; // equivalent to i / 16
      int index = i & 15; // equivalent to i % 16
      return (data[which] & (1 << index));
    }

    SYRAH_FORCEINLINE void set(int i, bool val) {
      int which = i >> 4;
      int index = i & 15;
      // set data[which, index] to val
      int on_bit = (1 << index);
      int reverse = ~on_bit;
      int current = data[which];
      current &= reverse;
      current |= ((val) ? on_bit : 0);
      data[which] = current;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator|(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_vkor(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator|=(const FixedVectorMask& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_vkor(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator&(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_vkand(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator&=(const FixedVectorMask& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_vkand(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator^(const FixedVectorMask& v) const {
      FixedVectorMask result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_vkxor(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask& operator^=(const FixedVectorMask& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_vkxor(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE static FixedVectorMask True() {
      FixedVectorMask result;
      __mmask all_on = _mm512_int2mask(~0x0);
      SYRAH_LRB_LOOP(i) {
        result.data[i] = all_on;
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask False() {
      FixedVectorMask result;
      __mmask all_off = _mm512_int2mask(0x0);
      SYRAH_LRB_LOOP(i) {
        result.data[i] = all_off;
      }
      return result;
    }

    SYRAH_FORCEINLINE static FixedVectorMask FirstN(int n) {
      __m512i n_splat = _mm512_set_1to16_pi(n);
      // UGH(boulos): Really?
      __m512i small_seq = _mm512_set_16to16_pi(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
      __m512i inc_val = _mm512_set_1to16_pi(n);
      FixedVectorMask result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmplt_pi(small_seq, n_splat);
        small_seq = _mm512_add_pi(small_seq, inc_val);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask operator!() const {
      FixedVectorMask result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_vknot(data[i]);
      }
      return result;
    }

    __mmask data[N/16];
  };

  template<int N>
  SYRAH_FORCEINLINE bool All(const FixedVectorMask<N, true>& v) {
    __mmask all_on = _mm512_int2mask(~0x0);
    SYRAH_LRB_LOOP(i) {
      if (v.data[i] != all_on) return false;
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE bool Any(const FixedVectorMask<N, true>& v) {
    __mmask zero = _mm512_int2mask(0);
    SYRAH_LRB_LOOP(i) {
      if (v.data[i] != zero) return true;
    }
    return false;
  }

  template<int N>
  SYRAH_FORCEINLINE bool None(const FixedVectorMask<N, true>& v) {
    __mmask zero = _mm512_int2mask(0);
    SYRAH_LRB_LOOP(i) {
      if (v.data[i] != zero) return false;
    }
    return true;
  }

  template<int N>
  SYRAH_FORCEINLINE int NumActive(const FixedVectorMask<N, true>& v) {
    int sum = 0;
    SYRAH_LRB_LOOP(i) {
      sum += _mm_countbits_16(static_cast<unsigned short>(v.data[i]));
    }
    return sum;
  }
#undef SYRAH_LRB_LOOP
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_LRB_MASK_H_
