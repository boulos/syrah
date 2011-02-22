#ifndef _SYRAH_FIXED_VECTOR_LRB_FLOAT_H_
#define _SYRAH_FIXED_VECTOR_LRB_FLOAT_H_

namespace syrah {
  template<int N>
  class SYRAH_ALIGN(64) FixedVector<float, N, true> {
  public:
#define SYRAH_LRB_LOOP(index) SYRAH_UNROLL(N/16) \
for (int index = 0; index < N/16; index++)

    SYRAH_FORCEINLINE FixedVector() {}
    SYRAH_FORCEINLINE FixedVector(const float value) {
      load(value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values) {
      load(values);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, bool /*aligned*/) {
      load_aligned(values);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, const FixedVectorMask<N>& mask,
                                 const float default_value) {
      load(values, mask, default_value);
    }

    SYRAH_FORCEINLINE FixedVector(const float* values, const FixedVectorMask<N>& mask,
                                 const float default_value, bool /*aligned*/) {
      load_aligned(values, mask, default_value);
    }


    SYRAH_FORCEINLINE FixedVector(const float* base, const FixedVector<int, N, true>& offsets, const int scale) {
      gather(base, offsets, scale);
    }

    SYRAH_FORCEINLINE FixedVector(const float* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
      gather(base, offsets, scale, mask);
    }

#if 0
    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<double, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm_set_ps(v[4*i + 3], v[4*i+2], v[4*i+1], v[4*i+0]);
      }
    }
#endif

    // Have to wait until <int> is ready.
    SYRAH_FORCEINLINE explicit FixedVector(const FixedVector<int, N> &v);
    static SYRAH_FORCEINLINE FixedVector<float, N, true> reinterpret(const FixedVector<int, N, true>& v);

    static SYRAH_FORCEINLINE FixedVector<float, N, true> Zero() {
      FixedVector<float, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_setzero_ps();
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector(const FixedVector<float, N, true>& v) {
      // TODO(boulos): Optimize this
      SYRAH_LRB_LOOP(i) {
        data[i] = v.data[i];
      }
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator=(const FixedVector<float, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = v.data[i];
      }
      return *this;
    }

    SYRAH_FORCEINLINE const float& operator[](int i) const {
      return ((float*)data)[i];
    }

    SYRAH_FORCEINLINE float& operator[](int i) {
      return ((float*)data)[i];
    }

    SYRAH_FORCEINLINE void load(const float value) {
      __m512 splat = _mm512_set_1to16_ps(value);
      SYRAH_LRB_LOOP(i) {
        data[i] = splat;
      }
    }

    SYRAH_FORCEINLINE void load(const float* values) {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(values, 64);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_loadd(const_cast<float*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_loadu_ps(const_cast<float*>(&(values[16*i])));
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const float* values) {
       SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_loadd(const_cast<float*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE);
       }
    }

    SYRAH_FORCEINLINE void load(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(values, 64);
      __m512 default_data = _mm512_set_1to16_ps(default_value);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_mask_loadd(default_data, mask.data[i], const_cast<float*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE);
        }
      } else {
        // TODO(boulos): If N is long enough, this could probably be
        // improved.
        SYRAH_LRB_LOOP(i) {
          data[i] = _mm512_mask_loadu_ps(const_cast<float*>(&(values[16*i])), mask.data[i], default_data);
        }
      }
    }

    SYRAH_FORCEINLINE void load_aligned(const float* values, const FixedVectorMask<N>& mask, const float default_value) {
      __m512 default_data = _mm512_set_1to16_ps(default_value);
      SYRAH_LRB_LOOP(i) {
         data[i] = _mm512_mask_loadd(default_data, mask.data[i], const_cast<float*>(&(values[16*i])), _MM_FULLUPC_NONE, _MM_BROADCAST_16X16, _MM_HINT_NONE);
      }
    }

    SYRAH_FORCEINLINE void gather(const float* base, const FixedVector<int, N, true>& offsets, const int scale) {
#define SYRAH_GATHER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { data[i] = _mm512_gatherd(offsets.data[i], const_cast<float*>(base), _MM_FULLUPC_NONE, SCALE, _MM_HINT_NONE); }
      switch (scale) {
      case 1: SYRAH_GATHER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_GATHER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_GATHER_LOOP(_MM_SCALE_4); break;
      default: SYRAH_GATHER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_GATHER_LOOP
    }

    // QUESTION(boulos): Constant scale only?
    SYRAH_FORCEINLINE void gather(const float* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) {
#define SYRAH_GATHER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { data[i] = _mm512_mask_gatherd(data[i], mask.data[i], offsets.data[i], const_cast<float*>(base), _MM_FULLUPC_NONE, SCALE, _MM_HINT_NONE); }
      switch (scale) {
      case 1: SYRAH_GATHER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_GATHER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_GATHER_LOOP(_MM_SCALE_4); break;
      default: SYRAH_GATHER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_GATHER_LOOP
    }

    SYRAH_FORCEINLINE void store(float* dst) const {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(dst, 64);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          _mm512_stored(&(dst[16*i]), data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
        }
      } else {
        SYRAH_LRB_LOOP(i) {
          _mm512_storeu_ps(&(dst[16*i]), data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst) const {
       SYRAH_LRB_LOOP(i) {
          _mm512_stored(&(dst[16*i]), data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
       }
    }

    SYRAH_FORCEINLINE void store_aligned_stream(float* dst) const {
       SYRAH_LRB_LOOP(i) {
          _mm512_stored(&(dst[16*i]), data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NT);
       }
    }

    SYRAH_FORCEINLINE void store(float* dst, const FixedVectorMask<N>& mask) const {
      bool lrb_aligned = SYRAH_IS_ALIGNED_POW2(dst, 64);
      if (lrb_aligned) {
        SYRAH_LRB_LOOP(i) {
          _mm512_mask_stored(&(dst[16*i]), mask.data[i], data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
        }
      } else {
        SYRAH_LRB_LOOP(i) {
          _mm512_mask_storeu_ps(&(dst[16*i]), mask.data[i], data[i]);
        }
      }
    }

    SYRAH_FORCEINLINE void store_aligned(float* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_LRB_LOOP(i) {
         _mm512_mask_stored(&(dst[16*i]), mask.data[i], data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NONE);
      }
    }

    SYRAH_FORCEINLINE void store_aligned_stream(float* dst, const FixedVectorMask<N>& mask) const {
      SYRAH_LRB_LOOP(i) {
         _mm512_mask_stored(&(dst[16*i]), mask.data[i], data[i], _MM_DOWNC_NONE, _MM_SUBSET32_16, _MM_HINT_NT);
      }
    }

    SYRAH_FORCEINLINE void scatter(float* base, const FixedVector<int, N, true>& offsets, const int scale) const {
#define SYRAH_SCATTER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { _mm512_scatterd(base, offsets.data[i], data[i], _MM_DOWNC_NONE, SCALE, _MM_HINT_NONE); }
      switch (scale) {
      case 1: SYRAH_SCATTER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_SCATTER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_SCATTER_LOOP(_MM_SCALE_4); break;
      default : SYRAH_SCATTER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_SCATTER_LOOP
    }

    // QUESTION(boulos): Constant scale only?
    SYRAH_FORCEINLINE void scatter(float* base, const FixedVector<int, N, true>& offsets, const int scale, const FixedVectorMask<N>& mask) const {
#define SYRAH_SCATTER_LOOP(SCALE) SYRAH_LRB_LOOP(i) { _mm512_mask_scatterd(base, mask.data[i], offsets.data[i], data[i], _MM_DOWNC_NONE, SCALE, _MM_HINT_NONE); }
      switch (scale) {
      case 1: SYRAH_SCATTER_LOOP(_MM_SCALE_1); break;
      case 2: SYRAH_SCATTER_LOOP(_MM_SCALE_2); break;
      case 4: SYRAH_SCATTER_LOOP(_MM_SCALE_4); break;
      default : SYRAH_SCATTER_LOOP(_MM_SCALE_8); break;
      }
#undef SYRAH_SCATTER_LOOP
    }

    SYRAH_FORCEINLINE void merge(const FixedVector<float, N, true>& v,
                                const FixedVectorMask<N, true>& mask) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_mask_movd(data[i], mask.data[i], v.data[i]);
      }
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator+(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_add_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator+=(const FixedVector<float, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_add_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator-() const {
      FixedVector<float, N, true> result;
      const __m512 sign_mask = _mm512_set_1to16_ps(-0.f);
      SYRAH_LRB_LOOP(i) {
        // XXX(boulos): Really? No xor_ps?
        //result.data[i] = _mm512_xor_ps(data[i], sign_mask);
        result.data[i] = _mm512_castsi512_ps(_mm512_xor_pi(_mm512_castps_si512(data[i]),
                                                           _mm512_castps_si512(sign_mask)));
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator-(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_sub_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator-=(const FixedVector<float, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_sub_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator*(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_mul_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator*=(const FixedVector<float, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_mul_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true> operator/(const FixedVector<float, N, true>& v) const {
      FixedVector<float, N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_div_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVector<float, N, true>& operator/=(const FixedVector<float, N, true>& v) {
      SYRAH_LRB_LOOP(i) {
        data[i] = _mm512_div_ps(data[i], v.data[i]);
      }
      return *this;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmplt_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator <=(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmple_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator ==(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        result.data[i] = _mm512_cmpeq_ps(data[i], v.data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >=(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        // No ge available, so a >= b <=> b <= a
        result.data[i] = _mm512_cmple_ps(v.data[i], data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE FixedVectorMask<N, true> operator >(const FixedVector<float, N, true>& v) const {
      FixedVectorMask<N, true> result;
      SYRAH_LRB_LOOP(i) {
        // No gt available, so do lt the other way. a > b <=> b < a
        result.data[i] = _mm512_cmplt_ps(v.data[i], data[i]);
      }
      return result;
    }

    SYRAH_FORCEINLINE float MinElement() const {
       __m512 min_vec = data[0];
       for (int i = 1; i < N/16; i++) {
          min_vec = _mm512_min_ps(data[i], min_vec);
       }
       return _mm512_reduce_min_ps(min_vec);
    }

    SYRAH_FORCEINLINE float MaxElement() const {
       __m512 max_vec = data[0];
       for (int i = 1; i < N/16; i++) {
          max_vec = _mm512_max_ps(data[i], max_vec);
       }
       return _mm512_reduce_max_ps(max_vec);
    }

    SYRAH_FORCEINLINE float foldMin() const { return MinElement(); }
    SYRAH_FORCEINLINE float foldMax() const { return MaxElement(); }
    SYRAH_FORCEINLINE float foldSum() const {
      __m512 result = _mm512_setzero_ps();
      SYRAH_LRB_LOOP(i) {
        result = _mm512_add_ps(result, data[i]);
      }
      return _mm512_reduce_add_ps(result);
    }

    SYRAH_FORCEINLINE float foldProd() const {
      __m512 result = _mm512_set_1to16_ps(1.f);
      SYRAH_LRB_LOOP(i) {
        result = _mm512_mul_ps(result, data[i]);
      }
      return _mm512_reduce_mul_ps(result);
    }

    __m512 data[N/16];
  };

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> max(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_max_ps(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> min(const FixedVector<float, N, true>& v1, const FixedVector<float, N, true>& v2) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_min_ps(v1.data[i], v2.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> sqrt(const FixedVector<float, N, true>& v1) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_sqrt_ps(v1.data[i]);
    }
    return result;
  }

  // a * b + c
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> madd(const FixedVector<float, N, true>& a,
                                                    const FixedVector<float, N, true>& b,
                                                    const FixedVector<float, N, true>& c) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_madd213_ps(a.data[i], b.data[i], c.data[i]);
    }
    return result;
  }

  // result = (float)((int)a)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> trunc(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
       result.data[i] = _mm512_round_ps(a.data[i], _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
    }
    return result;
  }

  // result = round_to_float(a) (using current rounding mode)
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> rint(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
       result.data[i] = _mm512_round_ps(a.data[i], _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
    }
    return result;
  }

  // Makes sense to override floor and ceil for LRB
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> floor(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_floor_ps(a.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> ceil(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_ceil_ps(a.data[i]);
    }
    return result;
  }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> sin(const FixedVector<float, N, true>& a) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_sin_ps(a.data[i]);
    }
    return result;
  }

  // output = (mask[i]) ? a : b
  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N, true> select(const FixedVector<float, N, true>& a,
                                                      const FixedVector<float, N, true>& b,
                                                      const FixedVectorMask<N>& mask) {
    FixedVector<float, N, true> result;
    SYRAH_LRB_LOOP(i) {
      result.data[i] = _mm512_mask_movd(b.data[i], mask.data[i], a.data[i]);
    }
    return result;
  }
} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_LRB_FLOAT_H_
