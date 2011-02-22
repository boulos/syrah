#include "syrah/FixedVector.h"
#include "syrah/CycleTimer.h"
#include <cmath>
#include <typeinfo>
#include <cstdlib>
#include <cstdio>
using namespace syrah;

static const int VecWidth = 16;
static const int kDefaultNumElements = 1024;
static const int kNumWarmup = 16;
static const int kNumBench  = 1024;

template <typename T>
bool ValueEquals(const T val, const T expected) {
  T epsilon(0);
  if (typeid(T) == typeid(float)) epsilon = 1e-7f;
  if (typeid(T) == typeid(double)) epsilon = 1e-15;

  if (val == expected) return true;
  // Handle nan
  if (val != val && expected != expected) return true;
  // if -epsilon <= (val - expected) <= epsilon return true
  if (val <= (expected + epsilon) &&
      val >= (expected - epsilon))
    return true;
  return false;
}

float* randomFloats(int how_many, unsigned int seed) {
  float* result = new float[how_many];
  unsigned int lcg = seed;
  for (int i = 0; i < how_many; i++) {
    lcg = 1664525 * lcg + 1013904223;
    result[i] = -10.f + 20.f * (lcg / 4294967296.f);
  }
  return result;
}

#define SCALAR_UNARY_FUNC(func, name) \
  void name ## _scalar(float* input, float* output, int num_elements) {\
    for (int i = 0; i < num_elements; i++) { \
      output[i] = (func)(input[i]); \
    } }

#define VECTOR_UNARY_FUNC(func, name) \
  void name ## _vector(float* input, float* output, int num_elements) {\
    int vec_end = num_elements & (~(VecWidth - 1)); \
    for (int i = 0; i < vec_end; i+= VecWidth) { \
      FixedVector<float, VecWidth> x_vec(&input[i], true); \
      FixedVector<float, VecWidth> eval = (func)(x_vec); \
      eval.store_aligned(output + i); \
    } \
    if (vec_end != num_elements) { \
      FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end); \
      FixedVector<float, VecWidth> x_vec(&(input[vec_end]), true); \
      FixedVector<float, VecWidth> eval = (func)(x_vec); \
      eval.store_aligned(output + vec_end, store_mask); \
    } }

#define SCALAR_BINARY_FUNC(func, name) \
  void name ## _scalar(float* x, float* y, float* output, int num_elements) {    \
    for (int i = 0; i < num_elements; i++) { \
      output[i] = (func)(x[i], y[i]); \
    } }

#define VECTOR_BINARY_FUNC(func, name) \
  void name ## _vector(float* x, float* y, float* output, int num_elements) {    \
    int vec_end = num_elements & (~(VecWidth - 1)); \
    for (int i = 0; i < vec_end; i+= VecWidth) { \
      FixedVector<float, VecWidth> x_vec(&x[i], true); \
      FixedVector<float, VecWidth> y_vec(&y[i], true); \
      FixedVector<float, VecWidth> eval = (func)(x_vec, y_vec); \
      eval.store_aligned(output + i); \
    } \
    if (vec_end != num_elements) { \
      FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end); \
      FixedVector<float, VecWidth> x_vec(&(x[vec_end]), true); \
      FixedVector<float, VecWidth> y_vec(&(y[vec_end]), true); \
      FixedVector<float, VecWidth> eval = (func)(x_vec, y_vec); \
      eval.store_aligned(output + vec_end, store_mask); \
    } }

#define UNARY_FUNC(scalar_func, vector_func, name) \
  SCALAR_UNARY_FUNC((scalar_func), name); \
  VECTOR_UNARY_FUNC((vector_func), name);

#define BINARY_FUNC(scalar_func, vector_func, name) \
  SCALAR_BINARY_FUNC((scalar_func), name); \
  VECTOR_BINARY_FUNC((vector_func), name);

UNARY_FUNC(sinf, sin, sin);
UNARY_FUNC(cosf, cos, cos);
UNARY_FUNC(tanf, tan, tan);
UNARY_FUNC(expf, exp, exp);
UNARY_FUNC(logf, log, log);
UNARY_FUNC(atanf, atan, atan);

BINARY_FUNC(powf, pow, pow);
BINARY_FUNC(atan2f, atan2, atan2);

void sincos_scalar(float* x_vals, float* sin_x, float* cos_x, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    sin_x[i] = sinf(x_vals[i]);
    cos_x[i] = cosf(x_vals[i]);
  }
}

void sincos_vector(float* x_vals, float* sin_x, float* cos_x, int num_elements) {
  int vec_end = num_elements & (~(VecWidth - 1));
  for (int i = 0; i < vec_end; i += VecWidth) {
    FixedVector<float, VecWidth> x_vec(&x_vals[i], true);
    FixedVector<float, VecWidth> sin_vec;
    FixedVector<float, VecWidth> cos_vec;
    sincos(x_vec, sin_vec, cos_vec);
    sin_vec.store_aligned(sin_x + i);
    cos_vec.store_aligned(cos_x + i);
  }
  if (vec_end != num_elements) {
    FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end);
    FixedVector<float, VecWidth> x_vec(&x_vals[vec_end], true);
    FixedVector<float, VecWidth> sin_vec;
    FixedVector<float, VecWidth> cos_vec;
    sincos(x_vec, sin_vec, cos_vec);
    sin_vec.store_aligned(sin_x + vec_end, store_mask);
    cos_vec.store_aligned(cos_x + vec_end, store_mask);
  }
}

typedef void(*UnaryFunc)(float*, float*, int);
typedef void(*BinaryFunc)(float*, float*, float*, int);
typedef void(*SinCosFunc)(float*, float*, float*, int);

void BenchmarkUnary(UnaryFunc scalar_func, UnaryFunc vector_func, float* x_vals, float* scalar_res, float* vector_res, int num_elements, int warmup, int bench, bool verify, const char* opname) {
  // Warmup
  for (int i = 0; i < warmup; i++) {
    scalar_func(x_vals, scalar_res, num_elements);
  }
  // Now bench
  CycleTimer::SysClock start_scalar = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    scalar_func(x_vals, scalar_res, num_elements);
  }
  CycleTimer::SysClock end_scalar   = CycleTimer::currentTicks();
  CycleTimer::SysClock scalar_ticks = end_scalar - start_scalar;

  // Warmup vector
  for (int i = 0; i < warmup; i++) {
    vector_func(x_vals, vector_res, num_elements);
  }
  // Now bench
  CycleTimer::SysClock start_vector = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    vector_func(x_vals, vector_res, num_elements);
  }
  CycleTimer::SysClock end_vector   = CycleTimer::currentTicks();
  CycleTimer::SysClock vector_ticks = end_vector - start_vector;

  if (verify) {
    for (int i = 0; i < num_elements; i++) {
      if (!ValueEquals<float>(vector_res[i], scalar_res[i])) {
        fprintf(stderr, "%s(%f) %06d: gold = %f != vector = %f\n", opname, x_vals[i], i, scalar_res[i], vector_res[i]);
      }
    }
  }

  double scalar_time_avg = scalar_ticks * CycleTimer::secondsPerTick() / bench;
  double vector_time_avg = vector_ticks * CycleTimer::secondsPerTick() / bench;

  CycleTimer::SysClock scalar_ticks_per_element = scalar_ticks / (num_elements * bench);
  CycleTimer::SysClock vector_ticks_per_element = vector_ticks / (num_elements * bench);

  printf("%s scalar time: %lf (%4.2f million elements per second, %d %s per element)\n", opname, scalar_time_avg, (num_elements / scalar_time_avg) / 1e6f, int(scalar_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector time: %lf (%4.2f million elements per second, %d %s per element)\n", opname, vector_time_avg, (num_elements / vector_time_avg) / 1e6f, int(vector_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector speedup: %.2f\n", opname, scalar_time_avg/vector_time_avg);
}

void BenchmarkBinary(BinaryFunc scalar_func, BinaryFunc vector_func, float* x_vals, float* y_vals, float* scalar_res, float* vector_res, int num_elements, int warmup, int bench, bool verify, const char* opname) {
  // Warmup
  for (int i = 0; i < warmup; i++) {
    scalar_func(x_vals, y_vals, scalar_res, num_elements);
  }
  // Now bench
  CycleTimer::SysClock start_scalar = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    scalar_func(x_vals, y_vals, scalar_res, num_elements);
  }
  CycleTimer::SysClock end_scalar   = CycleTimer::currentTicks();
  CycleTimer::SysClock scalar_ticks = end_scalar - start_scalar;

  // Warmup vector
  for (int i = 0; i < warmup; i++) {
    vector_func(x_vals, y_vals, vector_res, num_elements);
  }
  // Now bench
  CycleTimer::SysClock start_vector = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    vector_func(x_vals, y_vals, vector_res, num_elements);
  }
  CycleTimer::SysClock end_vector   = CycleTimer::currentTicks();
  CycleTimer::SysClock vector_ticks = end_vector - start_vector;

  if (verify) {
    for (int i = 0; i < num_elements; i++) {
      if (!ValueEquals<float>(vector_res[i], scalar_res[i])) {
        fprintf(stderr, "%s(%f) %06d: gold = %f != vector = %f\n", opname, x_vals[i], i, scalar_res[i], vector_res[i]);
      }
    }
  }

  double scalar_time_avg = scalar_ticks * CycleTimer::secondsPerTick() / bench;
  double vector_time_avg = vector_ticks * CycleTimer::secondsPerTick() / bench;

  CycleTimer::SysClock scalar_ticks_per_element = scalar_ticks / (num_elements * bench);
  CycleTimer::SysClock vector_ticks_per_element = vector_ticks / (num_elements * bench);

  printf("%s scalar time: %lf (%4.2f million elements per second, %d %s per element)\n", opname, scalar_time_avg, (num_elements / scalar_time_avg) / 1e6f, int(scalar_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector time: %lf (%4.2f million elements per second, %d %s per element)\n", opname, vector_time_avg, (num_elements / vector_time_avg) / 1e6f, int(vector_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector speedup: %.2f\n", opname, scalar_time_avg/vector_time_avg);
}


void BenchmarkSinCos(SinCosFunc scalar_func, SinCosFunc vector_func, float* x_vals, float* scalar_res0, float* scalar_res1, float* vector_res0, float* vector_res1, int num_elements, int warmup, int bench, bool verify, const char* opname) {
  // Warmup
  for (int i = 0; i < warmup; i++) {
    scalar_func(x_vals, scalar_res0, scalar_res1, num_elements);
  }
  // Now bench
  CycleTimer::SysClock start_scalar = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    scalar_func(x_vals, scalar_res0, scalar_res1, num_elements);
  }
  CycleTimer::SysClock end_scalar   = CycleTimer::currentTicks();
  CycleTimer::SysClock scalar_ticks = end_scalar - start_scalar;

  // Warmup vector
  for (int i = 0; i < warmup; i++) {
    vector_func(x_vals, vector_res0, vector_res1, num_elements);
  }
  // Now bench
  CycleTimer::SysClock start_vector = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    vector_func(x_vals, vector_res0, vector_res1, num_elements);
  }
  CycleTimer::SysClock end_vector   = CycleTimer::currentTicks();
  CycleTimer::SysClock vector_ticks = end_vector - start_vector;

  if (verify) {
    for (int i = 0; i < num_elements; i++) {
      if (!ValueEquals<float>(vector_res0[i], scalar_res0[i])) {
        fprintf(stderr, "%s_0(%f) %06d: gold = %f != vector = %f\n", opname, x_vals[i], i, scalar_res0[i], vector_res0[i]);
      }
    }
    for (int i = 0; i < num_elements; i++) {
      if (!ValueEquals<float>(vector_res1[i], scalar_res1[i])) {
        fprintf(stderr, "%s_1(%f) %06d: gold = %f != vector = %f\n", opname, x_vals[i], i, scalar_res1[i], vector_res1[i]);
      }
    }
  }

  double scalar_time_avg = scalar_ticks * CycleTimer::secondsPerTick() / bench;
  double vector_time_avg = vector_ticks * CycleTimer::secondsPerTick() / bench;

  CycleTimer::SysClock scalar_ticks_per_element = scalar_ticks / (num_elements * bench);
  CycleTimer::SysClock vector_ticks_per_element = vector_ticks / (num_elements * bench);

  printf("%s scalar time: %lf (%4.2f million elements per second, %d %s per element)\n", opname, scalar_time_avg, (num_elements / scalar_time_avg) / 1e6f, int(scalar_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector time: %lf (%4.2f million elements per second, %d %s per element)\n", opname, vector_time_avg, (num_elements / vector_time_avg) / 1e6f, int(vector_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector speedup: %.2f\n", opname, scalar_time_avg/vector_time_avg);
}

void test_trans(int num_elements, bool verify) {
  syrah::DisableDenormals();
  printf("testing trig on %d-wide arrays\n", num_elements);

  float* x_vals = randomFloats(num_elements, 0xDEADBEEF);
  float* y_vals = randomFloats(num_elements, 0xCAFEBABE);
  float* scalar_res0 = new float[num_elements];
  float* scalar_res1 = new float[num_elements];
  float* vector_res0 = new float[num_elements];
  float* vector_res1 = new float[num_elements];

  BenchmarkUnary(sin_scalar, sin_vector, x_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Sin");
  BenchmarkUnary(cos_scalar, cos_vector, x_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Cos");
  BenchmarkSinCos(sincos_scalar, sincos_vector, x_vals, scalar_res0, scalar_res1, vector_res0, vector_res1, num_elements, kNumWarmup, kNumBench, verify, "SinCos");
  BenchmarkUnary(tan_scalar, tan_vector, x_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Tan");
  BenchmarkUnary(exp_scalar, exp_vector, x_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Exp");
  BenchmarkUnary(log_scalar, log_vector, x_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Ln");
  BenchmarkUnary(atan_scalar, atan_vector, x_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Atan");

  BenchmarkBinary(pow_scalar, pow_vector, x_vals, y_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Pow");
  BenchmarkBinary(atan2_scalar, atan2_vector, x_vals, y_vals, scalar_res0, vector_res0, num_elements, kNumWarmup, kNumBench, verify, "Atan2");
}

#if !(defined(__APPLE__) && defined(__ARM_NEON__)) || defined(__LRB__)
int main(int argc, char** argv) {

  bool verify = false;
  int num_elements = kDefaultNumElements;


  if (argc != 1) {
    num_elements = atoi(argv[1]);
    if (argc == 3) {
      verify = (atoi(argv[2]) != 0);
    } else if (argc > 3) {
      fprintf(stderr, "ERROR: Bad number of arguments. Usage: trig [num_elements] or trig num_elements verify_int\n");
      exit(-1);
    }
  }

  test_trans(num_elements, verify);
  return 0;
}
#endif
