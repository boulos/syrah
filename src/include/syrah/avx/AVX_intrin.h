#ifndef _SYRAH_AVX_INTRIN_H_
#define _SYRAH_AVX_INTRIN_H_

#include "../Preprocessor.h"
#include "../sse/SSE_intrin.h"

namespace syrah {

#define SYRAH_AVX_CMP_PS(name, imm) SYRAH_FORCEINLINE __m256 _mm256_cmp##name##_ps(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, (imm)); }
#define SYRAH_AVX_CMP_PD(name, imm) SYRAH_FORCEINLINE __m256d _mm256_cmp##name##_pd(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, (imm)); }

  SYRAH_AVX_CMP_PS(lt, 1);
  SYRAH_AVX_CMP_PS(le, 2);
  SYRAH_AVX_CMP_PS(eq, 0);
  SYRAH_AVX_CMP_PS(ge, 5);
  SYRAH_AVX_CMP_PS(gt, 6);
  SYRAH_AVX_CMP_PS(neq, 4);

  SYRAH_AVX_CMP_PD(lt, 1);
  SYRAH_AVX_CMP_PD(le, 2);
  SYRAH_AVX_CMP_PD(eq, 0);
  SYRAH_AVX_CMP_PD(ge, 5);
  SYRAH_AVX_CMP_PD(gt, 6);
  SYRAH_AVX_CMP_PD(neq, 4);

#undef SYRAH_AVX_CMP_PS
#undef SYRAH_AVX_CMP_PD

#if !defined(__SSE4_1__) && !defined(__SSE4_2__)
  SYRAH_FORCEINLINE __m256 _mm256_blendv_ps(__m256 false_val, __m256 true_val, __m256 mask) {
    return _mm256_or_ps(_mm256_and_ps(mask, true_val), _mm256_andnot_ps(mask, false_val));
  }

  SYRAH_FORCEINLINE __m256d _mm256_blendv_pd(__m256d false_val, __m256d true_val, __m256d mask) {
    return _mm256_or_pd(_mm256_and_pd(mask, true_val), _mm256_andnot_pd(mask, false_val));
  }
#endif

  SYRAH_FORCEINLINE float syrah_mm256_hmax_ps(__m256 t) {
    // compute max([a, b, c, d, e, f, g, h], [b, b, d, d, f, f, h, h])
    __m256 temp1 = _mm256_max_ps(t, _mm256_movehdup_ps(t));

    // now we have [ab, bb, cd, dd, ef, ff, gh, hh]
    // the useful bits are:
    // 0 (ab)
    // 2 (cd)
    // 4 (ef)
    // 6 (gh)


    // I want [ab, cd, ab, cd, ...] vs [ef, gh, ef, gh, ...]

    // unpacklo(a, b) -> [ a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5] ]
    // unpackhi(a, b) -> [ a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7] ]
    __m256 temp2 = _mm256_unpacklo_ps(temp1, temp1); // [ab, ab, xx, xx, ef, ef, xx, xx]
    __m256 temp3 = _mm256_unpackhi_ps(temp1, temp1); // [cd, cd, xx, xx, gh, gh, xx, xx]

    __m256 temp4 = _mm256_max_ps(temp2, temp3); // [abcd, abcd, xx, xx, efgh, efgh, xx, xx]
    // Just get efgh over to the 0 spot
    __m256 temp5 = _mm256_permute2f128_ps(temp4, temp4, 0x1); // [efgh, efgh, xx, xx, abcd, abcd, xx, xx]
    __m256 temp6 = _mm256_max_ps(temp4, temp5); // [abcdefgh, abcdefgh, xx, xx, abcdefgh, abcdefgh, xx, xx]
    return _mm_cvtss_f32(_mm256_castps256_ps128(temp6));
  }

  SYRAH_FORCEINLINE float syrah_mm256_hmin_ps(__m256 t) {
    // compute min([a, b, c, d, e, f, g, h], [b, b, d, d, f, f, h, h])
    __m256 temp1 = _mm256_min_ps(t, _mm256_movehdup_ps(t));

    // now we have [ab, bb, cd, dd, ef, ff, gh, hh]
    // the useful bits are:
    // 0 (ab)
    // 2 (cd)
    // 4 (ef)
    // 6 (gh)


    // I want [ab, cd, ab, cd, ...] vs [ef, gh, ef, gh, ...]

    // unpacklo(a, b) -> [ a[0], b[0], a[1], b[1], a[4], b[4], a[5], b[5] ]
    // unpackhi(a, b) -> [ a[2], b[2], a[3], b[3], a[6], b[6], a[7], b[7] ]
    __m256 temp2 = _mm256_unpacklo_ps(temp1, temp1); // [ab, ab, xx, xx, ef, ef, xx, xx]
    __m256 temp3 = _mm256_unpackhi_ps(temp1, temp1); // [cd, cd, xx, xx, gh, gh, xx, xx]

    __m256 temp4 = _mm256_min_ps(temp2, temp3); // [abcd, abcd, xx, xx, efgh, efgh, xx, xx]
    // Just get efgh over to the 0 spot
    __m256 temp5 = _mm256_permute2f128_ps(temp4, temp4, 0x1); // [efgh, efgh, xx, xx, abcd, abcd, xx, xx]
    __m256 temp6 = _mm256_min_ps(temp4, temp5); // [abcdefgh, abcdefgh, xx, xx, abcdefgh, abcdefgh, xx, xx]
    return _mm_cvtss_f32(_mm256_castps256_ps128(temp6));
  }

  SYRAH_FORCEINLINE double syrah_mm256_hmax_pd(__m256d t) {
    // compute max([a, b, c, d], [b, b, d, d])
    __m256d temp1 = _mm256_max_pd(t, _mm256_unpackhi_pd(t, t));

    // now we have [ab, bb, cd, dd]
    // the useful bits are:
    // 0 (ab)
    // 2 (cd)

    // NOTE(boulos): We can't get at cd via shuffle (shuffle works
    // only one the low half/high half for dp), so we need extract to
    // reach cd but then we can just use SSE anyway.
    __m128d abbb = _mm256_extractf128_pd(temp1, 0);
    __m128d cddd = _mm256_extractf128_pd(temp1, 1);
    return _mm_cvtsd_f64(_mm_max_pd(abbb, cddd));
  }

  SYRAH_FORCEINLINE double syrah_mm256_hmin_pd(__m256d t) {
    // compute min([a, b, c, d], [b, b, d, d])
    __m256d temp1 = _mm256_min_pd(t, _mm256_unpackhi_pd(t, t));

    // now we have [ab, bb, cd, dd]
    // the useful bits are:
    // 0 (ab)
    // 2 (cd)

    // NOTE(boulos): We can't get at cd via shuffle (shuffle works
    // only one the low half/high half for dp), so we need extract to
    // reach cd but then we can just use SSE anyway.
    __m128d abbb = _mm256_extractf128_pd(temp1, 0);
    __m128d cddd = _mm256_extractf128_pd(temp1, 1);
    return _mm_cvtsd_f64(_mm_min_pd(abbb, cddd));
  }

}; // end namespace syrah

#endif // _SYRAH_AVX_INTRIN_H_
