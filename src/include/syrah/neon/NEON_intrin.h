#ifndef _SYRAH_NEON_INTRIN_H_
#define _SYRAH_NEON_INTRIN_H_

namespace syrah {
  SYRAH_FORCEINLINE float32x4_t syrah_rcp(float32x4_t v) {
    float32x4_t inv_v = vrecpeq_f32(v);
    float32x4_t step_val = vrecpsq_f32(v, inv_v);
    float32x4_t inv_better = vmulq_f32(inv_v, step_val);
    float32x4_t step_two = vrecpsq_f32(v, inv_better);
    float32x4_t true_inv = vmulq_f32(inv_better, step_two);
    return true_inv;
  }
}

#endif // _SYRAH_NEON_INTRIN_H_
