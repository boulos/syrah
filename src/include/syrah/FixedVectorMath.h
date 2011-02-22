#ifndef _SYRAH_FIXED_VECTOR_MATH_H_
#define _SYRAH_FIXED_VECTOR_MATH_H_

#include "FixedVector.h"
#include <cmath>

namespace syrah {

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> abs(const FixedVector<ElemType, N, SIMDMultiple>& v) {
    FixedVector<ElemType, N, SIMDMultiple> result;
    for (int i = 0; i < N; i++) {
      result[i] = std::abs(v[i]);
    }
    return result;
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> frac(const FixedVector<ElemType, N, SIMDMultiple>& v) {
    // Compute v - int(v)
    FixedVector<ElemType, N, SIMDMultiple> frac_val(v - trunc(v));

    FixedVectorMask<N> frac_lt_0(frac_val < FixedVector<ElemType, N, SIMDMultiple>::Zero());
    // result = (frac < 0) ? frac + 1 : frac;
    return select(frac_val + FixedVector<ElemType, N, SIMDMultiple>(ElemType(1)),
                  frac_val,
                  frac_lt_0);
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> floor(const FixedVector<ElemType, N, SIMDMultiple>& v) {
    FixedVector<ElemType, N, SIMDMultiple> truncated(trunc(v));
    // If v was lt 0 then it rounded towards 0 (upwards), so we need
    // to subtract 1 unless it was_integer (v == truncated) yielding
    // just v < truncated (thanks to Chris Kulla).
    //FixedVectorMask<N> need_adjust = v < truncated;
    FixedVectorMask<N> need_adjust = (v < truncated);
    const FixedVector<ElemType, N, SIMDMultiple> one(ElemType(1.));
    return select(truncated - one, truncated, need_adjust);
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> ceil(const FixedVector<ElemType, N, SIMDMultiple>& v) {
    // ceil(val) = val + 1-frac(val)
    FixedVector<ElemType, N, SIMDMultiple> frac_result(frac(v));
    FixedVectorMask<N> frac_eq_0(frac_result == FixedVector<ElemType, N, SIMDMultiple>(ElemType(0)));

    // result = (frac == 0) ? v : v + 1 - frac(v);
    return select(v,
                  v + FixedVector<ElemType, N, SIMDMultiple>(ElemType(1)) - frac_result,
                  frac_eq_0);
  }

  // sin(x) approximation from A&S (backported from sincos!)
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> sin(const FixedVector<ElemType, N, SIMDMultiple>& x_full) {
    const FixedVector<ElemType, N, SIMDMultiple> pi_over_two_vec(ElemType(1.57079637050628662109375));
    const FixedVector<ElemType, N, SIMDMultiple> two_over_pi_vec(ElemType(0.636619746685028076171875));
    FixedVector<ElemType, N> scaled = x_full * two_over_pi_vec;
    FixedVector<ElemType, N> k_real = floor(scaled);
    FixedVector<int, N> k(k_real);

    //const FixedVector<int, N> one(1);
    //const FixedVector<int, N> three(3);

    // Reduced range version of x
    FixedVector<ElemType, N, SIMDMultiple> x = x_full - k_real * pi_over_two_vec;
    FixedVector<int, N, SIMDMultiple> k_mod4(k & 1);
    FixedVectorMask<N> sin_usecos = (k_mod4 == 1 | k_mod4 == 3);
    FixedVectorMask<N> flip_sign = (k_mod4 > 1);

    // These coefficients are from sollya with fpminimax(sin(x)/x, [|0, 2, 4, 6, 8, 10|], [|single...|], [0;Pi/2]);
    const FixedVector<ElemType, N> one_vec(ElemType(1.));
    const FixedVector<ElemType, N> sin_c2(ElemType(-0.16666667163372039794921875));
    const FixedVector<ElemType, N> sin_c4(ElemType(8.333347737789154052734375e-3));
    const FixedVector<ElemType, N> sin_c6(ElemType(-1.9842604524455964565277099609375e-4));
    const FixedVector<ElemType, N> sin_c8(ElemType(2.760012648650445044040679931640625e-6));
    const FixedVector<ElemType, N> sin_c10(ElemType(-2.50293279435709337121807038784027099609375e-8));

    const FixedVector<ElemType, N> cos_c2(ElemType(-0.5));
    const FixedVector<ElemType, N> cos_c4(ElemType(4.166664183139801025390625e-2));
    const FixedVector<ElemType, N> cos_c6(ElemType(-1.388833043165504932403564453125e-3));
    const FixedVector<ElemType, N> cos_c8(ElemType(2.47562347794882953166961669921875e-5));
    const FixedVector<ElemType, N> cos_c10(ElemType(-2.59630184018533327616751194000244140625e-7));

    FixedVector<ElemType, N> outside = select(one_vec, x, sin_usecos);
    FixedVector<ElemType, N> c2 = select(cos_c2, sin_c2, sin_usecos);
    FixedVector<ElemType, N> c4 = select(cos_c4, sin_c4, sin_usecos);
    FixedVector<ElemType, N> c6 = select(cos_c6, sin_c6, sin_usecos);
    FixedVector<ElemType, N> c8 = select(cos_c8, sin_c8, sin_usecos);
    FixedVector<ElemType, N> c10 = select(cos_c10, sin_c10, sin_usecos);

    FixedVector<ElemType, N> x2 = x * x;
    FixedVector<ElemType, N> formula = madd(x2, c10, c8);
    formula = madd(x2, formula, c6);
    formula = madd(x2, formula, c4);
    formula = madd(x2, formula, c2);
    formula = madd(x2, formula, one_vec);
    formula *= outside;

    formula.merge(-formula, flip_sign);
    return formula;
  }

  // cos(x) approximation from A&S (backported from sincos!)
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> cos(const FixedVector<ElemType, N, SIMDMultiple>& x_full) {
    const FixedVector<ElemType, N, SIMDMultiple> pi_over_two_vec(ElemType(1.57079637050628662109375));
    const FixedVector<ElemType, N, SIMDMultiple> two_over_pi_vec(ElemType(0.636619746685028076171875));
    FixedVector<ElemType, N> scaled = x_full * two_over_pi_vec;
    FixedVector<ElemType, N> k_real = floor(scaled);
    FixedVector<int, N> k(k_real);

    // Reduced range version of x
    FixedVector<ElemType, N> x = x_full - k_real * pi_over_two_vec;

    const FixedVector<int, N> one(1);
    const FixedVector<int, N> two(2);
    const FixedVector<int, N> three(3);

    FixedVector<int, N> k_mod4(k & three);
    FixedVectorMask<N> cos_usecos = (k_mod4 == FixedVector<int, N>::Zero() | k_mod4 == two);
    FixedVectorMask<N> flip_sign = (k_mod4 == one | k_mod4 == two);

    const FixedVector<ElemType, N> one_vec(ElemType(1.));
    const FixedVector<ElemType, N> sin_c2(ElemType(-0.16666667163372039794921875));
    const FixedVector<ElemType, N> sin_c4(ElemType(8.333347737789154052734375e-3));
    const FixedVector<ElemType, N> sin_c6(ElemType(-1.9842604524455964565277099609375e-4));
    const FixedVector<ElemType, N> sin_c8(ElemType(2.760012648650445044040679931640625e-6));
    const FixedVector<ElemType, N> sin_c10(ElemType(-2.50293279435709337121807038784027099609375e-8));

    const FixedVector<ElemType, N> cos_c2(ElemType(-0.5));
    const FixedVector<ElemType, N> cos_c4(ElemType(4.166664183139801025390625e-2));
    const FixedVector<ElemType, N> cos_c6(ElemType(-1.388833043165504932403564453125e-3));
    const FixedVector<ElemType, N> cos_c8(ElemType(2.47562347794882953166961669921875e-5));
    const FixedVector<ElemType, N> cos_c10(ElemType(-2.59630184018533327616751194000244140625e-7));

    FixedVector<ElemType, N> outside = select(one_vec, x, cos_usecos);
    FixedVector<ElemType, N> c2 = select(cos_c2, sin_c2, cos_usecos);
    FixedVector<ElemType, N> c4 = select(cos_c4, sin_c4, cos_usecos);
    FixedVector<ElemType, N> c6 = select(cos_c6, sin_c6, cos_usecos);
    FixedVector<ElemType, N> c8 = select(cos_c8, sin_c8, cos_usecos);
    FixedVector<ElemType, N> c10 = select(cos_c10, sin_c10, cos_usecos);

    FixedVector<ElemType, N> x2 = x * x;
    FixedVector<ElemType, N> formula = madd(x2, c10, c8);
    formula = madd(x2, formula, c6);
    formula = madd(x2, formula, c4);
    formula = madd(x2, formula, c2);
    formula = madd(x2, formula, one_vec);
    formula *= outside;

    formula.merge(-formula, flip_sign);
    return formula;
  }


  // sincos(x) approximation from A&S (copied and pasted from sine, but use_cos is k == 0 || k == 2)
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE void sincos(const FixedVector<ElemType, N, SIMDMultiple>& x_full, FixedVector<ElemType, N>& sin_result, FixedVector<ElemType, N>& cos_result) {
    const FixedVector<ElemType, N, SIMDMultiple> pi_over_two_vec(ElemType(1.57079637050628662109375));
    const FixedVector<ElemType, N, SIMDMultiple> two_over_pi_vec(ElemType(0.636619746685028076171875));
    FixedVector<ElemType, N> scaled = x_full * two_over_pi_vec;
    FixedVector<ElemType, N> k_real = floor(scaled);
    FixedVector<int, N> k(k_real);

    const FixedVector<int, N> one(1);
    const FixedVector<int, N> two(2);
    const FixedVector<int, N> three(3);


    // Reduced range version of x
    FixedVector<ElemType, N, SIMDMultiple> x = x_full - k_real * pi_over_two_vec;
    FixedVector<int, N, SIMDMultiple> k_mod4(k & three);
    FixedVectorMask<N> cos_usecos = (k_mod4 == FixedVector<int, N>::Zero() | k_mod4 == two);
    FixedVectorMask<N> sin_usecos = (k_mod4 == one | k_mod4 == three);
    FixedVectorMask<N> sin_flipsign = (k_mod4 > one);
    FixedVectorMask<N> cos_flipsign = (k_mod4 == one | k_mod4 == two);

    const FixedVector<ElemType, N> one_vec(ElemType(1.));
    const FixedVector<ElemType, N> sin_c2(ElemType(-0.16666667163372039794921875));
    const FixedVector<ElemType, N> sin_c4(ElemType(8.333347737789154052734375e-3));
    const FixedVector<ElemType, N> sin_c6(ElemType(-1.9842604524455964565277099609375e-4));
    const FixedVector<ElemType, N> sin_c8(ElemType(2.760012648650445044040679931640625e-6));
    const FixedVector<ElemType, N> sin_c10(ElemType(-2.50293279435709337121807038784027099609375e-8));

    const FixedVector<ElemType, N> cos_c2(ElemType(-0.5));
    const FixedVector<ElemType, N> cos_c4(ElemType(4.166664183139801025390625e-2));
    const FixedVector<ElemType, N> cos_c6(ElemType(-1.388833043165504932403564453125e-3));
    const FixedVector<ElemType, N> cos_c8(ElemType(2.47562347794882953166961669921875e-5));
    const FixedVector<ElemType, N> cos_c10(ElemType(-2.59630184018533327616751194000244140625e-7));

    FixedVector<ElemType, N> x2 = x * x;

    FixedVector<ElemType, N> sin_formula = madd(x2, sin_c10, sin_c8);
    FixedVector<ElemType, N> cos_formula = madd(x2, cos_c10, cos_c8);
    sin_formula = madd(x2, sin_formula, sin_c6);
    cos_formula = madd(x2, cos_formula, cos_c6);

    sin_formula = madd(x2, sin_formula, sin_c4);
    cos_formula = madd(x2, cos_formula, cos_c4);

    sin_formula = madd(x2, sin_formula, sin_c2);
    cos_formula = madd(x2, cos_formula, cos_c2);

    sin_formula = madd(x2, sin_formula, one_vec);
    cos_formula = madd(x2, cos_formula, one_vec);

    sin_formula *= x;

    sin_result = select(cos_formula, sin_formula, sin_usecos);
    cos_result = select(cos_formula, sin_formula, cos_usecos);

    sin_result.merge(-sin_result, sin_flipsign);
    cos_result.merge(-cos_result, cos_flipsign);
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> tan(const FixedVector<ElemType, N, SIMDMultiple>& x_full) {
    const FixedVector<ElemType, N, SIMDMultiple> pi_over_four_vec(ElemType(0.785398185253143310546875));
    const FixedVector<ElemType, N, SIMDMultiple> four_over_pi_vec(ElemType(1.27323949337005615234375));

    FixedVectorMask<N> x_lt_0 = x_full < FixedVector<ElemType, N, SIMDMultiple>::Zero();
    FixedVector<ElemType, N, SIMDMultiple> y = select(-x_full, x_full, x_lt_0);
    FixedVector<ElemType, N> scaled = y * four_over_pi_vec;

    FixedVector<ElemType, N> k_real = floor(scaled);
    FixedVector<int, N> k(k_real);

    FixedVector<ElemType, N, SIMDMultiple> x = y - k_real * pi_over_four_vec;

    const FixedVector<int, N> one(1);
    const FixedVector<int, N> two(2);
    const FixedVector<int, N> three(3);

    // if k & 1, x -= Pi/4
    FixedVectorMask<N> need_offset = (k & one) != FixedVector<int, N, SIMDMultiple>::Zero();
    x.merge(x - pi_over_four_vec, need_offset);

    // if k & 3 == (0 or 3) let z = tan_In...(y) otherwise z = -cot_In0To...
    FixedVector<int, N, SIMDMultiple> k_mod4 = k & three;
    FixedVectorMask<N> use_cotan = (k_mod4 == one) | (k_mod4 == two);

    const FixedVector<ElemType, N> one_vec(ElemType(1.0));

    const FixedVector<ElemType, N> tan_c2(ElemType(0.33333075046539306640625));
    const FixedVector<ElemType, N> tan_c4(ElemType(0.13339905440807342529296875));
    const FixedVector<ElemType, N> tan_c6(ElemType(5.3348250687122344970703125e-2));
    const FixedVector<ElemType, N> tan_c8(ElemType(2.46033705770969390869140625e-2));
    const FixedVector<ElemType, N> tan_c10(ElemType(2.892402000725269317626953125e-3));
    const FixedVector<ElemType, N> tan_c12(ElemType(9.500005282461643218994140625e-3));

    const FixedVector<ElemType, N> cot_c2(ElemType(-0.3333333432674407958984375));
    const FixedVector<ElemType, N> cot_c4(ElemType(-2.222204394638538360595703125e-2));
    const FixedVector<ElemType, N> cot_c6(ElemType(-2.11752182804048061370849609375e-3));
    const FixedVector<ElemType, N> cot_c8(ElemType(-2.0846328698098659515380859375e-4));
    const FixedVector<ElemType, N> cot_c10(ElemType(-2.548247357481159269809722900390625e-5));
    const FixedVector<ElemType, N> cot_c12(ElemType(-3.5257363606433500535786151885986328125e-7));

    FixedVector<ElemType, N> x2 = x * x;
    FixedVector<ElemType, N> z;
    if (All(use_cotan)) {
      FixedVector<ElemType, N> cot_val = madd(x2, cot_c12, cot_c10);
      cot_val = madd(x2, cot_val, cot_c8);
      cot_val = madd(x2, cot_val, cot_c6);
      cot_val = madd(x2, cot_val, cot_c4);
      cot_val = madd(x2, cot_val, cot_c2);
      cot_val = madd(x2, cot_val, one_vec);
      // The equation is for x * cot(x) but we need -x * cot(x) for the tan part.
      cot_val /= -x;
      z = cot_val;
    } else if (None(use_cotan)) {
      FixedVector<ElemType, N> tan_val = madd(x2, tan_c12, tan_c10);
      tan_val = madd(x2, tan_val, tan_c8);
      tan_val = madd(x2, tan_val, tan_c6);
      tan_val = madd(x2, tan_val, tan_c4);
      tan_val = madd(x2, tan_val, tan_c2);
      tan_val = madd(x2, tan_val, one_vec);
      // Equation was for tan(x)/x
      tan_val *= x;
      z = tan_val;
    } else {
      FixedVector<ElemType, N> x2 = x * x;
      FixedVector<ElemType, N> tan_val = madd(x2, tan_c12, tan_c10);
      tan_val = madd(x2, tan_val, tan_c8);
      tan_val = madd(x2, tan_val, tan_c6);
      tan_val = madd(x2, tan_val, tan_c4);
      tan_val = madd(x2, tan_val, tan_c2);
      tan_val = madd(x2, tan_val, one_vec);
      // Equation was for tan(x)/x
      tan_val *= x;
      FixedVector<ElemType, N> cot_val = madd(x2, cot_c12, cot_c10);
      cot_val = madd(x2, cot_val, cot_c8);
      cot_val = madd(x2, cot_val, cot_c6);
      cot_val = madd(x2, cot_val, cot_c4);
      cot_val = madd(x2, cot_val, cot_c2);
      cot_val = madd(x2, cot_val, one_vec);
      // The equation is for x * cot(x) but we need -x * cot(x) for the tan part.
      cot_val /= -x;
      z = select(cot_val, tan_val, use_cotan);
    }
    return select(-z, z, x_lt_0);
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> atan(const FixedVector<ElemType, N, SIMDMultiple>& x_full) {
    const FixedVector<ElemType, N, SIMDMultiple> pi_over_two_vec(ElemType(1.57079637050628662109375));
    const FixedVector<ElemType, N> one(ElemType(1.0));
    // atan(-x) = -atan(x) (so flip from negative to positive first)
    // if x > 1 -> atan(x) = Pi/2 - atan(1/x)
    FixedVectorMask<N> x_neg = x_full < FixedVector<ElemType, N>::Zero();
    FixedVector<ElemType, N> x_flipped = select(-x_full, x_full, x_neg);

    FixedVectorMask<N> x_gt_1 = x_flipped > one;
    FixedVector<ElemType, N> x = select(one/x_flipped, x_flipped, x_gt_1);

    // These coefficients approximate atan(x)/x
    const FixedVector<ElemType, N> atan_c0(ElemType(0.99999988079071044921875));
    const FixedVector<ElemType, N> atan_c2(ElemType(-0.3333191573619842529296875));
    const FixedVector<ElemType, N> atan_c4(ElemType(0.199689209461212158203125));
    const FixedVector<ElemType, N> atan_c6(ElemType(-0.14015688002109527587890625));
    const FixedVector<ElemType, N> atan_c8(ElemType(9.905083477497100830078125e-2));
    const FixedVector<ElemType, N> atan_c10(ElemType(-5.93664981424808502197265625e-2));
    const FixedVector<ElemType, N> atan_c12(ElemType(2.417283318936824798583984375e-2));
    const FixedVector<ElemType, N> atan_c14(ElemType(-4.6721356920897960662841796875e-3));

    FixedVector<ElemType, N> x2 = x * x;
    FixedVector<ElemType, N> result = madd(x2, atan_c14, atan_c12);
    result = madd(x2, result, atan_c10);
    result = madd(x2, result, atan_c8);
    result = madd(x2, result, atan_c6);
    result = madd(x2, result, atan_c4);
    result = madd(x2, result, atan_c2);
    result = madd(x2, result, atan_c0);
    result *= x;

    result.merge(pi_over_two_vec - result, x_gt_1);
    result.merge(-result, x_neg);
    return result;
  }

  // The std math definition of atan2(y, x)
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple>
  atan2(const FixedVector<ElemType, N, SIMDMultiple>& y, const FixedVector<ElemType, N, SIMDMultiple>& x) {
    const FixedVector<ElemType, N, SIMDMultiple> pi_vec(ElemType(3.1415927410125732421875));
    const FixedVector<ElemType, N, SIMDMultiple> pi_over_two_vec(ElemType(1.57079637050628662109375));
    // atan2(y, x) =
    //
    // atan2(y > 0, x = +-0) ->  Pi/2
    // atan2(y < 0, x = +-0) -> -Pi/2
    // atan2(y = +-0, x < +0) -> +-Pi
    // atan2(y = +-0, x >= +0) -> +-0
    //
    // atan2(y >= 0, x < 0) ->  Pi + atan(y/x)
    // atan2(y <  0, x < 0) -> -Pi + atan(y/x)
    // atan2(y, x > 0) -> atan(y/x)
    //
    // and then a bunch of code for dealing with infinities.
    FixedVector<ElemType, N> y_over_x = y/x;
    FixedVector<ElemType, N> atan_arg = atan(y_over_x);
    FixedVectorMask<N> x_lt_0 = x < FixedVector<ElemType, N>::Zero();
    FixedVectorMask<N> y_lt_0 = y < FixedVector<ElemType, N>::Zero();
    FixedVector<ElemType, N> offset = select(select(-pi_vec, pi_vec, y_lt_0), FixedVector<ElemType, N>::Zero(), x_lt_0);
    return offset + atan_arg;
  }

  // exp(x)
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> exp(const FixedVector<ElemType, N, SIMDMultiple>& x_full) {
#define SYRAH_EXP_CW 1
#if SYRAH_EXP_CW
    const FixedVector<ElemType, N> ln2_part1(ElemType(0.6931457519));
    const FixedVector<ElemType, N> ln2_part2(ElemType(1.4286067653e-6));
#else
    const FixedVector<ElemType, N> ln2(ElemType(0.693147182464599609375));
#endif
    const FixedVector<ElemType, N> one_over_ln2(ElemType(1.44269502162933349609375));

    FixedVector<ElemType, N> scaled = x_full * one_over_ln2;
    FixedVector<ElemType, N> k_real = floor(scaled);
    FixedVector<int, N> k(k_real);

    // Reduced range version of x
#if SYRAH_EXP_CW
    FixedVector<ElemType, N, SIMDMultiple> x = x_full - k_real * ln2_part1;
    x -= k_real * ln2_part2;
#else
    FixedVector<ElemType, N, SIMDMultiple> x = x_full - k_real * ln2;
#endif


    // These coefficients are for e^x in [0, ln(2)]
    const FixedVector<ElemType, N> one(ElemType(1.));
    const FixedVector<ElemType, N> c2(ElemType(0.4999999105930328369140625));
    const FixedVector<ElemType, N> c3(ElemType(0.166668415069580078125));
    const FixedVector<ElemType, N> c4(ElemType(4.16539050638675689697265625e-2));
    const FixedVector<ElemType, N> c5(ElemType(8.378830738365650177001953125e-3));
    const FixedVector<ElemType, N> c6(ElemType(1.304379315115511417388916015625e-3));
    const FixedVector<ElemType, N> c7(ElemType(2.7555381529964506626129150390625e-4));

    FixedVector<ElemType, N> result = madd(x, c7, c6);
    result = madd(x, result, c5);
    result = madd(x, result, c4);
    result = madd(x, result, c3);
    result = madd(x, result, c2);
    result = madd(x, result, one);
    result = madd(x, result, one);

    // Compute 2^k (should differ for float and double, but I'll avoid
    // it for now and just do floats)
    const FixedVector<int, N> fpbias(127);
    FixedVector<int, N> biased_n = k + fpbias;
    FixedVectorMask<N> overflow = k > fpbias;
    // Minimum exponent is -126, so if k is <= -127 (k + 127 <= 0)
    // we've got underflow. -127 * ln(2) -> -88.02. So the most
    // negative float input that doesn't result in zero is like -88.
    FixedVectorMask<N> underflow = biased_n <= FixedVector<int, N>::Zero();
    const FixedVector<int, N> InfBits = 0x7f800000;
    biased_n <<= 23;
    // Reinterpret this thing as float
    FixedVector<float, N> two_to_the_n = FixedVector<float, N>::reinterpret(biased_n);
    // Handle both doubles and floats (hopefully eliding the copy for float)
    FixedVector<ElemType, N> elemtype_2n(two_to_the_n);
    result *= elemtype_2n;
    result.merge(FixedVector<float, N>::reinterpret(InfBits), overflow);
    result.merge(FixedVector<float, N>::Zero(), underflow);
    return result;
  }

  // Range reduction for logarithms takes log(x) -> log(2^n * y) -> n
  // * log(2) + log(y) where y is the reduced range (usually in [1/2,
  // 1)).
  template<int N>
  SYRAH_FORCEINLINE void range_reduce_log(const FixedVector<float, N>& input, FixedVector<float, N>& reduced, FixedVector<int, N>& exponent) {
    FixedVector<int, N> int_version = FixedVector<int, N>::reinterpret(input);
    // single precision = SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM
    // exponent mask    = 0111 1111 1000 0000 0000 0000 0000 0000
    //                    0x7  0xF  0x8  0x0  0x0  0x0  0x0  0x0
    // non-exponent     = 1000 0000 0111 1111 1111 1111 1111 1111
    //                  = 0x8  0x0  0x7  0xF  0xF  0xF  0xF  0xF

    //const FixedVector<int, N> exponent_mask(0x7F800000);
    const FixedVector<int, N> nonexponent_mask(0x807FFFFF);

    // We want the reduced version to have an exponent of -1 which is -1 + 127 after biasing or 126
    const FixedVector<int, N> exponent_neg1(126 << 23);
    // NOTE(boulos): We don't need to mask anything out since we know
    // the sign bit has to be 0. If it's 1, we need to return infinity/nan
    // anyway (log(x), x = +-0 -> infinity, x < 0 -> NaN).
    FixedVector<int, N> biased_exponent = int_version >> 23; // This number is [0, 255] but it means [-127, 128]
    const FixedVector<int, N> one(1);
    const FixedVector<int, N> one_twenty_seven(127);

    FixedVector<int, N> offset_exponent = biased_exponent + one; // Treat the number as if it were 2^{e+1} * (1.m)/2
    exponent = offset_exponent - one_twenty_seven; // get the real value

    // Blend the offset_exponent with the original input (do this in
    // int for now, until I decide if float can have & and &not)
    FixedVector<int, N> blended = (int_version & nonexponent_mask) | (exponent_neg1);
#if 0
    std::cout << "input = " << input << std::endl;
    std::cout << "biased_exp = " << biased_exponent << std::endl;
    std::cout << "true_exp = " << exponent << std::endl;
    std::cout << "blended = " << blended << std::endl;
    std::cout << "reduced = " << reduced << std::endl;
#endif
    reduced = FixedVector<float, N>::reinterpret(blended);
  }

  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> ln(const FixedVector<ElemType, N, SIMDMultiple>& x_full) {
    // Get the exponent out of x_full
    FixedVector<ElemType, N> reduced;
    FixedVector<int, N> exponent;
    // NOTE(boulos): Again, these need to be long long int for doubles...
    const FixedVector<int, N> NaN_bits = 0x7fc00000;
    const FixedVector<int, N> Neg_Inf_bits = 0xFF800000;
    const FixedVector<ElemType, N> NaN = FixedVector<ElemType, N>::reinterpret(NaN_bits);
    const FixedVector<ElemType, N> neg_inf = FixedVector<ElemType, N>::reinterpret(Neg_Inf_bits);
    FixedVectorMask<N> use_nan = x_full < FixedVector<ElemType, N>::Zero();
    FixedVectorMask<N> use_inf = x_full == FixedVector<ElemType, N>::Zero();
    FixedVectorMask<N> exceptional = use_nan | use_inf;
    const FixedVector<ElemType, N> one(ElemType(1.0));
    // NOTE(boulos): Avoid generating NaNs which can be slower to
    // handle on some hardware. ln(1) should be friendly.
    FixedVector<ElemType, N> patched = select(one, x_full, exceptional);
    range_reduce_log(patched, reduced, exponent);

    const FixedVector<ElemType, N> ln2(ElemType(0.693147182464599609375));

    FixedVector<float, N> x1 = one - reduced;
    const FixedVector<ElemType, N> c1(ElemType(0.50000095367431640625));
    const FixedVector<ElemType, N> c2(ElemType(0.33326041698455810546875));
    const FixedVector<ElemType, N> c3(ElemType(0.2519190013408660888671875));
    const FixedVector<ElemType, N> c4(ElemType(0.17541764676570892333984375));
    const FixedVector<ElemType, N> c5(ElemType(0.3424419462680816650390625));
    const FixedVector<ElemType, N> c6(ElemType(-0.599632322788238525390625));
    const FixedVector<ElemType, N> c7(ElemType(+1.98442304134368896484375));
    const FixedVector<ElemType, N> c8(ElemType(-2.4899270534515380859375));
    const FixedVector<ElemType, N> c9(ElemType(+1.7491014003753662109375));

    FixedVector<ElemType, N> result = madd(x1, c9, c8);
    result = madd(x1, result, c7);
    result = madd(x1, result, c6);
    result = madd(x1, result, c5);
    result = madd(x1, result, c4);
    result = madd(x1, result, c3);
    result = madd(x1, result, c2);
    result = madd(x1, result, c1);
    result = madd(x1, result, one);

    // Equation was for -(ln(red)/(1-red))
    result *= -x1;
    result += (FixedVector<float, N>(exponent) * ln2);

    return select(select(NaN, neg_inf, use_nan), result, exceptional);
  }

  // The cmath name
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> log(const FixedVector<ElemType, N, SIMDMultiple>& x) {
    return ln(x);
  }

  // Compute x ^ y
  template<typename ElemType, int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<ElemType, N, SIMDMultiple> pow(const FixedVector<ElemType, N, SIMDMultiple>& x, const FixedVector<ElemType, N, SIMDMultiple>& y) {
    return exp(y * ln(x));
  }

  template<int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<float, N, SIMDMultiple> sqrtf(const FixedVector<float, N, SIMDMultiple>& v1) { return sqrt(v1); }

  template<int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<float, N, SIMDMultiple> rintf(const FixedVector<float, N, SIMDMultiple>& v1) { return rint(v1); }

  template<int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<float, N, SIMDMultiple> floorf(const FixedVector<float, N, SIMDMultiple>& v1) { return floor(v1); }

  template<int N, bool SIMDMultiple>
  SYRAH_FORCEINLINE FixedVector<float, N, SIMDMultiple> ceilf(const FixedVector<float, N, SIMDMultiple>& v1) { return ceil(v1); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> fabsf(const FixedVector<float, N>& x) { return abs(x); }
  template<int N>
  SYRAH_FORCEINLINE FixedVector<double, N> fabs(const FixedVector<double, N>& x) { return abs(x); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> sinf(const FixedVector<float, N>& x) { return sin(x); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> cosf(const FixedVector<float, N>& x) { return cos(x); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> tanf(const FixedVector<float, N>& x) { return tan(x); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> atanf(const FixedVector<float, N>& x) { return atan(x); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> expf(const FixedVector<float, N>& x) { return exp(x); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> logf(const FixedVector<float, N>& x) { return ln(x); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> powf(const FixedVector<float, N>& x, const FixedVector<float, N>& y) { return pow(x, y); }

  template<int N>
  SYRAH_FORCEINLINE FixedVector<float, N> atan2f(const FixedVector<float, N>& x, const FixedVector<float, N>& y) { return atan2(x, y); }


} // end namespace syrah

#endif // _SYRAH_FIXED_VECTOR_MATH_H_
