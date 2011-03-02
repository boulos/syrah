#ifndef _SYRAH_FIXED_VECTOR_AVX_H_
#define _SYRAH_FIXED_VECTOR_AVX_H_

// Get AVX

#ifdef __AVX__
#include <immintrin.h>
#else
#include "../external/avxintrin_emu.h"
#endif
#include "../Preprocessor.h"

#include "../sse/SSE_intrin.h"
#include "AVX_intrin.h"

#include "FixedVector_AVX_mask.h"
#include "FixedVector_AVX_float.h"
#include "FixedVector_AVX_double.h"
#include "FixedVector_AVX_int.h"

#include "FixedVector_AVX_casts.h"

#endif // _SYRAH_FIXED_VECTOR_AVX_H_
