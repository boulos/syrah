#include "syrah/FixedVector.h"
#include "syrah/CycleTimer.h"
#include <cstdio>
using namespace syrah;
static const int VecWidth = 16;
float* randomFloats(int how_many, unsigned int seed) {
  float* result = new float[how_many];
  unsigned int lcg = seed;
  for (int i = 0; i < how_many; i++) {
    lcg = 1664525 * lcg + 1013904223;
    result[i] = lcg / 4294967296.f;
  }
  return result;
}

void saxpy_scalar(float* scales, float* x_vals, float* y_vals, float* results, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    results[i] = scales[i] * x_vals[i] + y_vals[i];
  }
}

void saxpy_unrolled(float* scales, float* x_vals, float* y_vals, float* results, int num_elements) {
  int vec_end = num_elements & (~(VecWidth - 1));
  for (int i = 0; i < vec_end; i+= VecWidth) {
    results[i +  0] = scales[i +  0] * x_vals[i +  0] + y_vals[i +  0];
    results[i +  1] = scales[i +  1] * x_vals[i +  1] + y_vals[i +  1];
    results[i +  2] = scales[i +  2] * x_vals[i +  2] + y_vals[i +  2];
    results[i +  3] = scales[i +  3] * x_vals[i +  3] + y_vals[i +  3];

    results[i +  4] = scales[i +  4] * x_vals[i +  4] + y_vals[i +  4];
    results[i +  5] = scales[i +  5] * x_vals[i +  5] + y_vals[i +  5];
    results[i +  6] = scales[i +  6] * x_vals[i +  6] + y_vals[i +  6];
    results[i +  7] = scales[i +  7] * x_vals[i +  7] + y_vals[i +  7];

    results[i +  8] = scales[i +  8] * x_vals[i +  8] + y_vals[i +  8];
    results[i +  9] = scales[i +  9] * x_vals[i +  9] + y_vals[i +  9];
    results[i + 10] = scales[i + 10] * x_vals[i + 10] + y_vals[i + 10];
    results[i + 11] = scales[i + 11] * x_vals[i + 11] + y_vals[i + 11];

    results[i + 12] = scales[i + 12] * x_vals[i + 12] + y_vals[i + 12];
    results[i + 13] = scales[i + 13] * x_vals[i + 13] + y_vals[i + 13];
    results[i + 14] = scales[i + 14] * x_vals[i + 14] + y_vals[i + 14];
    results[i + 15] = scales[i + 15] * x_vals[i + 15] + y_vals[i + 15];
  }
  if (vec_end != num_elements) {
    for (int i = vec_end; i < num_elements; i++) {
      results[i] = scales[i] * x_vals[i] + y_vals[i];
    }
  }
}

void saxpy_vector(float* scales, float* x_vals, float* y_vals, float* results, int num_elements) {
  int vec_end = num_elements & (~(VecWidth - 1));
  for (int i = 0; i < vec_end; i += VecWidth) {
    FixedVector<float, VecWidth> scale_vec(&scales[i], true);
    FixedVector<float, VecWidth> x_vec(&x_vals[i], true);
    FixedVector<float, VecWidth> y_vec(&y_vals[i], true);
    FixedVector<float, VecWidth> result = madd(scale_vec, x_vec, y_vec);
    result.store_aligned(results + i);
  }
  if (vec_end != num_elements) {
    int i = vec_end;
    FixedVectorMask<VecWidth> mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end);
    FixedVector<float, VecWidth> scale_vec(&scales[i], true);
    FixedVector<float, VecWidth> x_vec(&x_vals[i], true);
    FixedVector<float, VecWidth> y_vec(&y_vals[i], true);
    FixedVector<float, VecWidth> result = madd(scale_vec, x_vec, y_vec);
    result.store_aligned(results + i, mask);
  }
}

int main() {
  syrah::DisableDenormals();
  // NOTE(boulos): For big vectors, the speeds will be similar as it's
  // purely L2 bound. For smaller ones, we hit in cache and then it's
  // mostly dependent on L1-cache performance and math.
  static const int kNumElements = 1024;
  static const int kNumWarmup = 16;
  static const int kNumBench  = 1024;
  float* scales = randomFloats(kNumElements, 0xCAFEBABE);
  float* x_vals = randomFloats(kNumElements, 0xDEADBEEF);
  float* y_vals = randomFloats(kNumElements, 0xDEADBABE);
  float* gold_results = new float[kNumElements];
  float* vector_results = new float[kNumElements];

  // Warmup gold
  for (int i = 0; i < kNumWarmup; i++) {
    saxpy_scalar(scales, x_vals, y_vals, gold_results, kNumElements);
    //saxpy_unrolled(scales, x_vals, y_vals, gold_results, kNumElements);
  }
  // Measure gold
  CycleTimer::SysClock start_scalar = CycleTimer::currentTicks();
  for (int i = 0; i < kNumBench; i++) {
    saxpy_scalar(scales, x_vals, y_vals, gold_results, kNumElements);
    //saxpy_unrolled(scales, x_vals, y_vals, gold_results, kNumElements);
  }
  CycleTimer::SysClock end_scalar = CycleTimer::currentTicks();

  // Warmup vector
  for (int i = 0; i < kNumWarmup; i++) {
    saxpy_vector(scales, x_vals, y_vals, vector_results, kNumElements);
  }
  // Measure vector
  CycleTimer::SysClock start_vector = CycleTimer::currentTicks();
  for (int i = 0; i < kNumBench; i++) {
    saxpy_vector(scales, x_vals, y_vals, vector_results, kNumElements);
  }
  CycleTimer::SysClock end_vector = CycleTimer::currentTicks();

  // Verify results
  for (int i = 0; i < kNumElements; i++) {
    if (gold_results[i] != vector_results[i]) {
      printf("%06d: gold = %f != vector = %f\n", i, gold_results[i], vector_results[i]);
    }
  }

  CycleTimer::SysClock scalar_ticks = end_scalar - start_scalar;
  CycleTimer::SysClock vector_ticks = end_vector - start_vector;

  double scalar_time_avg = scalar_ticks * CycleTimer::secondsPerTick() / kNumBench;
  double vector_time_avg = vector_ticks * CycleTimer::secondsPerTick() / kNumBench;

  CycleTimer::SysClock scalar_tpe = scalar_ticks / (kNumElements * kNumBench);
  CycleTimer::SysClock vector_tpe = vector_ticks / (kNumElements * kNumBench);

  printf("scalar time: %lf (%4.2f million elements per second, %d %s per element)\n", scalar_time_avg, kNumElements / scalar_time_avg / 1e6f, int(scalar_tpe), CycleTimer::tickUnits());
  printf("vector time: %lf (%4.2f million elements per second, %d %s per element)\n", vector_time_avg, kNumElements / vector_time_avg / 1e6f, int(vector_tpe), CycleTimer::tickUnits());
  printf("speedup:   %.2f\n", scalar_time_avg/vector_time_avg);
  return 0;
}
