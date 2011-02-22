#ifndef _SYRAH_LRB_INTRIN_H_
#define _SYRAH_LRB_INTRIN_H_

#include "../Preprocessor.h"

namespace syrah {
  // replace compressd with vloadu on LRB.
  SYRAH_FORCEINLINE void DisableDenormals() {
    // Figure out how to do this on lrb
  }

  SYRAH_FORCEINLINE __m512 _mm512_loadu_size4(const void* values) {
    // UGH(boulos): Stupid LRB prototype thing didn't have const.
    return _mm512_expandd(const_cast<void*>(values), _MM_FULLUPC_NONE, _MM_HINT_NONE);
  }

  SYRAH_FORCEINLINE __m512 _mm512_mask_loadu_size4(const void* values, __mmask mask, __m512 orig_val) {
    return _mm512_mask_expandd(orig_val, mask, const_cast<void*>(values), _MM_FULLUPC_NONE, _MM_HINT_NONE);
  }

  SYRAH_FORCEINLINE void _mm512_storeu_size4(void* values, __m512 v) {
    _mm512_compressd(values, v, _MM_DOWNC_NONE, _MM_HINT_NONE);
  }

  // These masked versions are tricky... Not sure if these are right.
  SYRAH_FORCEINLINE void _mm512_mask_storeu_size4(void* values, __mmask mask, __m512 v) {
    _mm512_mask_compressd(values, mask, v, _MM_DOWNC_NONE, _MM_HINT_NONE);
  }

  SYRAH_FORCEINLINE __m512 _mm512_loadu_ps(const float* values) {
    return _mm512_loadu_size4(values);
  }

  SYRAH_FORCEINLINE __m512i _mm512_loadu_epi32(const int* values) {
    return _mm512_castps_si512(_mm512_loadu_size4(values));
  }

  SYRAH_FORCEINLINE __m512 _mm512_mask_loadu_ps(const float* values, __mmask mask, __m512 v) {
    return _mm512_mask_loadu_size4(values, mask, v);
  }

  SYRAH_FORCEINLINE __m512i _mm512_mask_loadu_epi32(const int* values, __mmask mask, __m512i v) {
    return _mm512_castps_si512(_mm512_mask_loadu_size4(values, mask, _mm512_castsi512_ps(v)));
  }

  SYRAH_FORCEINLINE void _mm512_storeu_ps(float* values, __m512 v) {
    return _mm512_storeu_size4(values, v);
  }
  SYRAH_FORCEINLINE void _mm512_storeu_epi32(int* values, __m512i v) {
    return _mm512_storeu_size4(values, _mm512_castsi512_ps(v));
  }

  SYRAH_FORCEINLINE void _mm512_mask_storeu_ps(float* values, __mmask mask, __m512 v) {
    return _mm512_mask_storeu_size4(values, mask, v);
  }
  SYRAH_FORCEINLINE void _mm512_mask_storeu_epi32(int* values, __mmask mask, __m512i v) {
    return _mm512_mask_storeu_size4(values, mask, _mm512_castsi512_ps(v));
  }

  // NOTE(boulos): the external header had a terrible name.
  SYRAH_FORCEINLINE int _mm512_reduce_min_epi32(__m512i a) {
    return _mm512_reduce_min_pi(a);
  }

  SYRAH_FORCEINLINE int _mm512_reduce_max_epi32(__m512i a) {
    return _mm512_reduce_max_pi(a);
  }
}

#endif
