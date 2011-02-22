#ifndef _SYRAH_FIXED_VECTOR_SSE_H_
#define _SYRAH_FIXED_VECTOR_SSE_H_

// Get SSE3+
#ifndef __SSE3__
#error "SYRAH requires at least SSE3, so you can have ints"
#endif

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
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

#endif // _SYRAH_FIXED_VECTOR_SSE_H_