#ifndef _SYRAH_FIXED_VECTOR_NEON_H_
#define _SYRAH_FIXED_VECTOR_NEON_H_

// Get NEON
#include <arm_neon.h>
#include "../Preprocessor.h"

namespace syrah {
    inline void DisableDenormals() {
    }
};

#include "FixedVector_NEON_mask.h"
#include "FixedVector_NEON_float.h"
//#include "FixedVector_NEON_double.h"
#include "FixedVector_NEON_int.h"
#include "FixedVector_NEON_uint8.h"
#include "FixedVector_NEON_casts.h"

#endif // _SYRAH_FIXED_VECTOR_NEON_H_
