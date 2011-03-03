#ifndef _SYRAH_FIXED_VECTOR_SSE_H_
#define _SYRAH_FIXED_VECTOR_SSE_H_

// Get SSE3+
#ifndef __SSE2__
#error "SYRAH requires at least SSE2, so you can have ints"
#endif

#include <xmmintrin.h>
#include <emmintrin.h>

#ifdef __SSE3__
#include <pmmintrin.h>
#endif

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#if defined(__SSE4_2__) || defined(__SSE4_1__)
#include <smmintrin.h>
#endif

#include "../Preprocessor.h"

#include "SSE_intrin.h"

#include "FixedVector_SSE_mask.h"
#include "FixedVector_SSE_float.h"
#include "FixedVector_SSE_double.h"
#include "FixedVector_SSE_int.h"
#include "FixedVector_SSE_casts.h"

#endif // _SYRAH_FIXED_VECTOR_SSE_H_
