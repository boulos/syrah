#ifndef _SYRAH_SSE_INTRIN_H_
#define _SYRAH_SSE_INTRIN_H_

#include "../Preprocessor.h"

namespace syrah {

  SYRAH_FORCEINLINE void DisableDenormals() {
#ifndef _MM_DENORM_ZERO_ON
#define _MM_DENORM_ZERO_ON 0x0040
#endif
    // Disables denormal handling and set flush to zero
    int oldMXCSR = _mm_getcsr(); //read the old MXCSR setting
    int newMXCSR = oldMXCSR | _MM_FLUSH_ZERO_ON | _MM_DENORM_ZERO_ON; // set DAZ and FZ bits
    _mm_setcsr( newMXCSR ); //write the new MXCSR setting to the MXCSR
  }

  // NOTE(boulos): Fuck SSE. I'm going to make blend true, false, mask.
  SYRAH_FORCEINLINE __m128 syrah_blendv_ps(__m128 true_val, __m128 false_val, __m128 mask) {
#if defined(__SSE4_1__)
    return _mm_blendv_ps(false_val, true_val, mask);
#else
    return _mm_or_ps(_mm_and_ps(mask, true_val), _mm_andnot_ps(mask, false_val));
#endif
  }

  SYRAH_FORCEINLINE __m128d syrah_blendv_pd(__m128d true_val, __m128d false_val, __m128d mask) {
#if defined(__SSE4_1__)
    return _mm_blendv_pd(false_val, true_val, mask);
#else
    return _mm_or_pd(_mm_and_pd(mask, true_val), _mm_andnot_pd(mask, false_val));
#endif
  }

  SYRAH_FORCEINLINE __m128i syrah_blendv_int(__m128i true_val, __m128i false_val, __m128i mask) {
#if defined(__SSE4_1__)
    // NOTE(boulos): Even though it's byte-level this works for all
    // values anyway.
    return _mm_blendv_epi8(false_val, true_val, mask);
#else
    return _mm_or_si128(_mm_and_si128(mask, true_val), _mm_andnot_si128(mask, false_val));
#endif
  }

  // Shamelessly lifted from Manta which lifted it from intel.com
  SYRAH_FORCEINLINE __m128i syrah_mm_mul_epi32(__m128i a, __m128i b) {
#if defined(__SSE4_1__)
    return _mm_mullo_epi32(a, b);
#else
    __m128i t0;
    __m128i t1;

    t0 = _mm_mul_epu32(a,b);
    t1 = _mm_mul_epu32( _mm_shuffle_epi32( a, 0xB1 ),
                        _mm_shuffle_epi32( b, 0xB1 ) );

    t0 = _mm_shuffle_epi32( t0, 0xD8 );
    t1 = _mm_shuffle_epi32( t1, 0xD8 );

    return _mm_unpacklo_epi32( t0, t1 );
#endif
  }

  SYRAH_FORCEINLINE __m128i syrah_mm_max_epi32(__m128i a, __m128i b) {
      __m128i a_gt_b = _mm_cmpgt_epi32(a, b);
      return syrah_blendv_int(a, b, a_gt_b);
  }

  SYRAH_FORCEINLINE __m128i syrah_mm_min_epi32(__m128i a, __m128i b) {
      __m128i a_lt_b = _mm_cmplt_epi32(a, b);
      return syrah_blendv_int(a, b, a_lt_b);
  }

  // Shamelessly lifted from Manta
  SYRAH_FORCEINLINE int syrah_mm_hmax_epi32(__m128i t) {
    // a = (t0, t0, t1, t1)
    __m128i a = _mm_unpacklo_epi32(t,t);
    // b = (t2, t2, t3, t3)
    __m128i b = _mm_unpackhi_epi32(t,t);
    // c = (max(t0,t2), max(t0, t2), max(t1, t3), max(t1, t3))
    __m128i c = syrah_mm_max_epi32(a, b);
    // The movehl will move the high 2 values to the low 2 values.
    // This will allow us to compare max(t0,t2) with max(t1, t3).
    __m128i max = syrah_mm_max_epi32(c, _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(c), _mm_castsi128_ps(c))));
    return _mm_cvtsi128_si32(max);
  }

  // Shamelessly lifted from Manta
  SYRAH_FORCEINLINE int syrah_mm_hmin_epi32(__m128i t) {
    // a = (t0, t0, t1, t1)
    __m128i a = _mm_unpacklo_epi32(t,t);
    // b = (t2, t2, t3, t3)
    __m128i b = _mm_unpackhi_epi32(t,t);
    // c = (max(t0,t2), max(t0, t2), max(t1, t3), max(t1, t3))
    __m128i c = syrah_mm_min_epi32(a, b);
    // The movehl will move the high 2 values to the low 2 values.
    // This will allow us to compare max(t0,t2) with max(t1, t3).
    __m128i max = syrah_mm_min_epi32(c, _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(c), _mm_castsi128_ps(c))));
    return _mm_cvtsi128_si32(max);
  }

  SYRAH_FORCEINLINE float syrah_mm_hmax_ps(__m128 t) {
    // compute max([a, b, c, d], [b, b, d, d])
    __m128 temp1 = _mm_max_ps(t, _mm_movehdup_ps(t));
    // now we have [max(a, b), max(b, b), max(c, d), max(d, d)], grab out max(c, d)
    __m128 shuffled = _mm_shuffle_ps(temp1, temp1, _MM_SHUFFLE(2, 2, 2, 2));
    // now max(max(a, b), max(c, d))
    __m128 all_max = _mm_max_ps(temp1, shuffled);
    return _mm_cvtss_f32(all_max);
  }

  SYRAH_FORCEINLINE float syrah_mm_hmin_ps(__m128 t) {
    // compute min([a, b, c, d], [b, b, d, d])
    __m128 temp1 = _mm_min_ps(t, _mm_movehdup_ps(t));
    // now we have [min(a, b), min(b, b), min(c, d), min(d, d)], grab out min(c, d)
    __m128 shuffled = _mm_shuffle_ps(temp1, temp1, _MM_SHUFFLE(2, 2, 2, 2));
    // now min(min(a, b), min(c, d))
    __m128 all_min = _mm_min_ps(temp1, shuffled);
    return _mm_cvtss_f32(all_min);
  }

  SYRAH_FORCEINLINE double syrah_mm_hmax_pd(__m128d t) {
    // compute max([a, b], [b, b])
    __m128d temp1 = _mm_max_pd(t, _mm_unpackhi_pd(t, t));
    return _mm_cvtsd_f64(temp1);
  }

  SYRAH_FORCEINLINE double syrah_mm_hmin_pd(__m128d t) {
    // compute min([a, b], [b, b])
    __m128d temp1 = _mm_min_pd(t, _mm_unpackhi_pd(t, t));
    return _mm_cvtsd_f64(temp1);
  }

}; // end namespace syrah

#endif // _SYRAH_SSE_INTRIN_H_
