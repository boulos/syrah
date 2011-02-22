#include "syrah/FixedVector.h"
#include "syrah/CycleTimer.h"
#include <cstdio>
using namespace syrah;

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

void saxpy_vector(float* scales, float* x_vals, float* y_vals, float* results, int num_elements) {
  static const int VecWidth = 32;
  int vec_end = num_elements & (~(VecWidth - 1));
  for (int i = 0; i < vec_end; i += VecWidth) {
    FixedVector<float, VecWidth> scale_vec(&scales[i], true);
    FixedVector<float, VecWidth> x_vec(&x_vals[i], true);
    FixedVector<float, VecWidth> y_vec(&y_vals[i], true);
    FixedVector<float, VecWidth> result = scale_vec * x_vec + y_vec;
    result.store_aligned(results + i);
  }
  if (vec_end != num_elements) {
    FixedVector<float, VecWidth> scale_vec(&scales[vec_end], true);
    FixedVector<float, VecWidth> x_vec(&x_vals[vec_end], true);
    FixedVector<float, VecWidth> y_vec(&y_vals[vec_end], true);
    FixedVector<float, VecWidth> result = scale_vec * x_vec + y_vec;
    FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end);
    result.store_aligned(results + vec_end, store_mask);
  }
}

int main() {
  syrah::DisableDenormals();

  static const int kNumElements = 1024 * 1024 * 16;
  static const int kNumWarmup = 4;
  static const int kNumBench  = 16;
  float* scales = randomFloats(kNumElements, 0xCAFEBABE);
  float* x_vals = randomFloats(kNumElements, 0xDEADBEEF);
  float* y_vals = randomFloats(kNumElements, 0xDEADBABE);
  float* gold_results = new float[kNumElements];
  float* vector_results = new float[kNumElements];

  // Warmup gold
  for (int i = 0; i < kNumWarmup; i++) {
    saxpy_scalar(scales, x_vals, y_vals, gold_results, kNumElements);
  }
  // Measure gold
  double start_gold = CycleTimer::currentSeconds();
  for (int i = 0; i < kNumBench; i++) {
    saxpy_scalar(scales, x_vals, y_vals, gold_results, kNumElements);
  }
  double end_gold = CycleTimer::currentSeconds();

  // Warmup vector
  for (int i = 0; i < kNumWarmup; i++) {
    saxpy_vector(scales, x_vals, y_vals, vector_results, kNumElements);
  }
  // Measure vector
  double start_vec = CycleTimer::currentSeconds();
  for (int i = 0; i < kNumBench; i++) {
    saxpy_vector(scales, x_vals, y_vals, vector_results, kNumElements);
  }
  double end_vec = CycleTimer::currentSeconds();

  // Verify results
  for (int i = 0; i < kNumElements; i++) {
    if (gold_results[i] != vector_results[i]) {
      printf("%06d: gold = %f != vector = %f\n", i, gold_results[i], vector_results[i]);
    }
  }

  double gold_time_avg = (end_gold - start_gold) / kNumBench;
  double vec_time_avg = (end_vec - start_vec) / kNumBench;
  printf("gold time: %lf\n", gold_time_avg);
  printf("vect time: %lf\n", vec_time_avg);
  printf("speedup:   %.2f\n", gold_time_avg/vec_time_avg);
  return 0;
}
