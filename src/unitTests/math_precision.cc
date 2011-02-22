#include <cstdio>
#include <cmath>
#include <gmp.h>
#include <mpfr.h>
#include "syrah/FixedVector.h"

using namespace syrah;
static const int VecWidth = 16;

float* randomFloats(int how_many, unsigned int seed) {
  float* result = new float[how_many];
  unsigned int lcg = seed;
  for (int i = 0; i < how_many; i++) {
    lcg = 1664525 * lcg + 1013904223;
    result[i] = (lcg / 4294967296.f);
  }
  return result;
}

#define MPFR_UNARY_FUNC(func, name, x_min, x_max)                             \
  void name ## _scalar(float* input, float* output, int num_elements) {\
    for (int i = 0; i < num_elements; i++) { \
      mpfr_t mpfr_input; \
      mpfr_init2(mpfr_input, 150); \
      mpfr_t mpfr_output; \
      mpfr_init2(mpfr_output, 150); \
      mpfr_set_flt(mpfr_input, (x_min) + input[i] * ((x_max) - (x_min)), MPFR_RNDN); \
      (func)(mpfr_output, mpfr_input, MPFR_RNDN); \
      output[i] = mpfr_get_flt(mpfr_output, MPFR_RNDN); \
    } }

#define VECTOR_UNARY_FUNC(func, name, x_min, x_max)                           \
  void name ## _vector(float* input, float* output, int num_elements) {\
    int vec_end = num_elements & (~(VecWidth - 1)); \
    const FixedVector<float, VecWidth> x0((float)x_min); \
    const FixedVector<float, VecWidth> x1((float)x_max); \
    for (int i = 0; i < vec_end; i+= VecWidth) { \
      FixedVector<float, VecWidth> x_input(&input[i], true); \
      FixedVector<float, VecWidth> x_vec = x0 + x_input * (x1-x0); \
      FixedVector<float, VecWidth> eval = (func)(x_vec); \
      eval.store_aligned(output + i); \
    } \
    if (vec_end != num_elements) { \
      FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end); \
      FixedVector<float, VecWidth> x_input(&(input[vec_end]), true); \
      FixedVector<float, VecWidth> x_vec = x0 + x_input * (x1-x0); \
      FixedVector<float, VecWidth> eval = (func)(x_vec); \
      eval.store_aligned(output + vec_end, store_mask); \
    } }

#define MPFR_BINARY_FUNC(func, name, x_min, x_max, y_min, y_max)              \
  void name ## _scalar(float* x, float* y, float* output, int num_elements) {    \
    for (int i = 0; i < num_elements; i++) { \
      mpfr_t mpfr_x; \
      mpfr_init2(mpfr_x, 150); \
      mpfr_t mpfr_y; \
      mpfr_init2(mpfr_y, 150); \
      mpfr_t mpfr_output; \
      mpfr_init2(mpfr_output, 150); \
      mpfr_set_flt(mpfr_x, (x_min) + x[i] * ((x_max) - (x_min)), MPFR_RNDN); \
      mpfr_set_flt(mpfr_y, (y_min) + y[i] * ((y_max) - (y_min)), MPFR_RNDN); \
      (func)(mpfr_output, mpfr_x, mpfr_y, MPFR_RNDN);  \
      output[i] = mpfr_get_flt(mpfr_output, MPFR_RNDN); \
    } }

#define VECTOR_BINARY_FUNC(func, name, x_min, x_max, y_min, y_max)            \
  void name ## _vector(float* x, float* y, float* output, int num_elements) {    \
    int vec_end = num_elements & (~(VecWidth - 1)); \
    const FixedVector<float, VecWidth> x0((float)x_min); \
    const FixedVector<float, VecWidth> x1((float)x_max); \
    const FixedVector<float, VecWidth> y0((float)y_min); \
    const FixedVector<float, VecWidth> y1((float)y_max); \
    for (int i = 0; i < vec_end; i+= VecWidth) { \
      FixedVector<float, VecWidth> x_input(&x[i], true); \
      FixedVector<float, VecWidth> y_input(&y[i], true); \
      FixedVector<float, VecWidth> x_vec = x0 + x_input * (x1-x0); \
      FixedVector<float, VecWidth> y_vec = y0 + y_input * (y1-y0); \
      FixedVector<float, VecWidth> eval = (func)(x_vec, y_vec); \
      eval.store_aligned(output + i); \
    } \
    if (vec_end != num_elements) { \
      FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end); \
      FixedVector<float, VecWidth> x_input(&(x[vec_end]), true); \
      FixedVector<float, VecWidth> y_input(&(y[vec_end]), true); \
      FixedVector<float, VecWidth> x_vec = x0 + x_input * (x1-x0); \
      FixedVector<float, VecWidth> y_vec = y0 + y_input * (y1-y0); \
      FixedVector<float, VecWidth> eval = (func)(x_vec, y_vec); \
      eval.store_aligned(output + vec_end, store_mask); \
    } }

#define UNARY_FUNC(scalar_func, vector_func, name, x_min, x_max)  \
   MPFR_UNARY_FUNC((scalar_func), name, x_min, x_max);                    \
   VECTOR_UNARY_FUNC((vector_func), name, x_min, x_max);

#define BINARY_FUNC(scalar_func, vector_func, name, x_min, x_max, y_min, y_max) \
   MPFR_BINARY_FUNC((scalar_func), name, x_min, x_max, y_min, y_max);         \
   VECTOR_BINARY_FUNC((vector_func), name, x_min, x_max, y_min, y_max);


struct ApproximationError {
   float avg_error;
   float max_error; float max_error_input; float max_error_binary_input;
  float max_error_expected; float max_error_computed;
   float avg_rel_error;
   float max_rel_error; float max_rel_error_input; float max_rel_error_binary_input;
  float max_rel_error_expected; float max_rel_error_computed;
   float avg_ulp_error;
   int max_ulp_error; float max_ulp_error_input; float max_ulp_error_binary_input;
  float max_ulp_error_expected; float max_ulp_error_computed;
};

void ComputeError(float* gold_version, float* compare_version, float* input, float x_min, float x_max, float* inputB, float y_min, float y_max, int num_elements, ApproximationError& error,
                  const char* func_name, float normalization, const char* norm_units, bool verbose) {
   memset(&error, 0, sizeof(ApproximationError));
   for (int i = 0; i < num_elements; i++) {
      struct IntFloat {
      public:
        union {
          float f;
          int i;
        };
      };

      IntFloat gold_union, compare_union;
      gold_union.f = gold_version[i];
      compare_union.f = compare_version[i];

      int gold_int = (gold_union.i < 0) ? (0x80000000 - gold_union.i) : gold_union.i;
      int approx_int = (compare_union.i < 0) ? (0x80000000 - compare_union.i) : compare_union.i;
      int ulpdiff = std::abs(gold_int - approx_int);
      float abs_diff = fabsf(gold_union.f - compare_union.f);
      if (std::isinf(gold_union.f) && std::isinf(compare_union.f)) abs_diff = 0.f;

      float rel_diff = abs_diff/fabsf(gold_union.f);
      error.avg_error += abs_diff;
      error.avg_ulp_error += (float)ulpdiff;
      error.avg_rel_error += rel_diff;

      float actual_input = x_min + input[i] * (x_max - x_min);
      float second_input = (inputB) ? y_min + inputB[i] * (y_max - y_min) : 0.f;

      if (abs_diff > error.max_error) {
         error.max_error = abs_diff;
         error.max_error_input = actual_input;
         if (inputB) error.max_error_binary_input = second_input;
         error.max_error_expected = gold_union.f;
         error.max_error_computed = compare_union.f;
      }
      if (rel_diff > error.max_rel_error) {
         error.max_rel_error = rel_diff;
         error.max_rel_error_input = actual_input;
         if (inputB) error.max_rel_error_binary_input = second_input;
         error.max_rel_error_expected = gold_union.f;
         error.max_rel_error_computed = compare_union.f;
      }
      if (ulpdiff > error.max_ulp_error) {
         error.max_ulp_error = ulpdiff;
         error.max_ulp_error_input = actual_input;
         if (inputB) error.max_ulp_error_binary_input = second_input;
         error.max_ulp_error_expected = gold_union.f;
         error.max_ulp_error_computed = compare_union.f;
      }

      if (verbose) {
         float normalized_input = actual_input / normalization;
         printf("%s(%14.8g ~ %14.8g%s): gold_result = %14.8g (0x%x), approx_result = %14.8g (0x%x), abs_diff = %14.8g, rel_diff = %14.8g, ulps = %d\n", func_name, actual_input, normalized_input, norm_units, gold_union.f, gold_union.i, compare_union.f, compare_union.i, abs_diff, rel_diff, ulpdiff);
      }
   }
   error.avg_rel_error /= num_elements;
   error.avg_ulp_error /= num_elements;
   error.avg_error /= num_elements;
   // Print the summary
   printf("%8s: Avg Error          %14.8g, Avg Rel Error %14.8g, Avg Ulp Error %14.8g\n", func_name, error.avg_error, error.avg_rel_error, error.avg_ulp_error);
   if (inputB) {
     printf("%8s: Max Error          %14.8g (for input %14.8g -> %14.8g%s, %14.8g -> %14.8g%s got %14.8g but expected %14.8g)\n", func_name, error.max_error, error.max_error_input, error.max_error_input / normalization, norm_units, error.max_error_binary_input, error.max_error_binary_input / normalization, norm_units, error.max_error_computed, error.max_error_expected);
      printf("%8s: Max Relative Error %14.8g (for input %14.8g -> %14.8g%s, %14.8g -> %14.8g%s got %14.8g but expected %14.8g)\n", func_name, error.max_rel_error, error.max_rel_error_input, error.max_rel_error_input / normalization, norm_units, error.max_rel_error_binary_input, error.max_rel_error_binary_input / normalization, norm_units, error.max_rel_error_computed, error.max_rel_error_expected);
      printf("%8s: Max Ulp Error      %14d (for input %14.8g -> %14.8g%s, %14.8g -> %14.8g%s got %14.8g but expected %14.8g)\n", func_name, error.max_ulp_error, error.max_ulp_error_input, error.max_ulp_error_input / normalization, norm_units, error.max_ulp_error_binary_input, error.max_ulp_error_binary_input / normalization, norm_units, error.max_ulp_error_computed, error.max_ulp_error_expected);
   } else {
      printf("%8s: Max Error          %14.8g (for input %14.8g -> %14.8g%s got %14.8g but expected %14.8g)\n", func_name, error.max_error, error.max_error_input, error.max_error_input / normalization, norm_units, error.max_error_computed, error.max_error_expected);
      printf("%8s: Max Relative Error %14.8g (for input %14.8g -> %14.8g%s got %14.8g but expected %14.8g)\n", func_name, error.max_rel_error, error.max_rel_error_input, error.max_rel_error_input / normalization, norm_units, error.max_rel_error_computed, error.max_rel_error_expected);
      printf("%8s: Max Ulp Error      %14d (for input %14.8g -> %14.8g%s got %14.8g but expected %14.8g)\n", func_name, error.max_ulp_error, error.max_ulp_error_input, error.max_ulp_error_input / normalization, norm_units, error.max_ulp_error_computed, error.max_ulp_error_expected);
   }
   printf("\n");
}

#define TRIG_MIN -10.f * M_PI
#define TRIG_MAX  10.f * M_PI
#define EXP_MIN -90.f
#define EXP_MAX  90.f
#define LOG_MIN  .1f
#define LOG_MAX  1024 * 1024
#define ATAN_MIN -1.f
#define ATAN_MAX  1.f

#define POW_XMIN .1f
#define POW_XMAX 100.f
#define POW_YMIN -4096.f
#define POW_YMAX  4096.f

#define ATAN2_XMIN -10.f
#define ATAN2_XMAX  10.f
#define ATAN2_YMIN -10.f
#define ATAN2_YMAX  10.f

UNARY_FUNC(mpfr_sin, sin, sin, TRIG_MIN, TRIG_MAX);
UNARY_FUNC(mpfr_cos, cos, cos, TRIG_MIN, TRIG_MAX);
UNARY_FUNC(mpfr_tan, tan, tan, TRIG_MIN, TRIG_MAX);
UNARY_FUNC(mpfr_exp, exp, exp, EXP_MIN, EXP_MAX);
UNARY_FUNC(mpfr_log, log, log, LOG_MIN, LOG_MAX);
UNARY_FUNC(mpfr_atan, atan, atan, ATAN_MIN, ATAN_MAX);

BINARY_FUNC(mpfr_pow, pow, pow, POW_XMIN, POW_XMAX, POW_YMIN, POW_YMAX);
BINARY_FUNC(mpfr_atan2, atan2, atan2, ATAN2_XMIN, ATAN2_XMAX, ATAN2_YMIN, ATAN2_YMAX);


#define COMPARE_UNARY_FUNCTION(func_name, x_min, x_max, norm, units) do { \
      func_name ## _scalar (x_vals, gold_results, num_elements); \
      func_name ## _vector (x_vals, approx_results, num_elements); \
      ApproximationError error_struct; \
      ComputeError(gold_results, approx_results, x_vals, x_min, x_max, NULL, 0, 0, num_elements, error_struct, #func_name, norm, units, print_all); \
   } while (0);

#define COMPARE_BINARY_FUNCTION(func_name, x_min, x_max, y_min, y_max, norm, units) do { \
      func_name ## _scalar (x_vals, y_vals, gold_results, num_elements); \
      func_name ## _vector (x_vals, y_vals, approx_results, num_elements); \
      ApproximationError error_struct; \
      ComputeError(gold_results, approx_results, x_vals, x_min, x_max, y_vals, y_min, y_max, num_elements, error_struct, #func_name, norm, units, print_all); \
   } while (0);

int main() {
  syrah::DisableDenormals();
  int num_elements = 2048 * 100;
  bool print_all = false;
  float* x_vals = randomFloats(num_elements, 0xDEADBEEF);
  float* y_vals = randomFloats(num_elements, 0xCAFEBABE);
  std::sort(x_vals, x_vals + num_elements);
  float* gold_results = new float[num_elements];
  float* approx_results = new float[num_elements];

  COMPARE_UNARY_FUNCTION(sin, TRIG_MIN, TRIG_MAX, M_PI_2, " Pi/2");
  COMPARE_UNARY_FUNCTION(cos, TRIG_MIN, TRIG_MAX, M_PI_2, " Pi/2");
  COMPARE_UNARY_FUNCTION(tan, TRIG_MIN, TRIG_MAX, M_PI_2, " Pi/2");
  COMPARE_UNARY_FUNCTION(exp, EXP_MIN,  EXP_MAX,  M_LN2, " Ln2");
  COMPARE_UNARY_FUNCTION(log, LOG_MIN,  LOG_MAX,  M_LN2, " Ln2");
  COMPARE_UNARY_FUNCTION(atan, ATAN_MIN, ATAN_MAX, 1.f, "");

  COMPARE_BINARY_FUNCTION(pow, POW_XMIN, POW_XMAX, POW_YMIN, POW_YMAX, 1.f, "");
  COMPARE_BINARY_FUNCTION(atan2, ATAN2_XMIN, ATAN2_XMAX, ATAN2_YMIN, ATAN2_YMAX, 1.f, "");

  return 0;
}
