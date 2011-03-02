#include "syrah/FixedVector.h"
#include "syrah/CycleTimer.h"
using namespace syrah;
static const int VecWidth = 16;
static const int kDefaultNumElements = 1024;
static const int kNumWarmup = 16;
static const int kNumBench  = 1024 * 16 * 16;

float* randomFloats(int how_many, unsigned int seed) {
  float* result = new float[how_many];
  unsigned int lcg = seed;
  for (int i = 0; i < how_many; i++) {
    lcg = 1664525 * lcg + 1013904223;
    result[i] = -10.f + 20.f * (lcg / 4294967296.f);
  }
  return result;
}

void gather_scalar(const float* input, float* output, const int* permute, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = input[permute[i]];
  }
}

void copy_scalar(const float* input, float* output, const int* permute, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = input[i];
  }
}

void copy_scalar_unroll16(const float* input, float* output, const int* permute, int num_elements) {
  int vec_end = num_elements & (~(VecWidth-1));
  for (int i = 0; i < vec_end; i+= VecWidth) {
#if 0
    float* dst = &output[i];
    const float* src = &input[i];
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;

    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;

    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;

    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
    *dst++ = *src++;
#else
    output[i + 0] = input[i + 0];
    output[i + 1] = input[i + 1];
    output[i + 2] = input[i + 2];
    output[i + 3] = input[i + 3];

    output[i + 4] = input[i + 4];
    output[i + 5] = input[i + 5];
    output[i + 6] = input[i + 6];
    output[i + 7] = input[i + 7];

    output[i + 8] = input[i + 8];
    output[i + 9] = input[i + 9];
    output[i + 10] = input[i + 10];
    output[i + 11] = input[i + 11];

    output[i + 12] = input[i + 12];
    output[i + 13] = input[i + 13];
    output[i + 14] = input[i + 14];
    output[i + 15] = input[i + 15];
#endif
  }
  if (vec_end != num_elements) {
    for (int i = vec_end; i < num_elements; i++) {
      output[i] = input[i];
    }
  }
}

void memcpy_scalar(const float* input, float* output, const int* permute, int num_elements) {
  memcpy(output, input, sizeof(float) * num_elements);
}

void reverse_scalar(const float* input, float* output, const int* permute, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = input[num_elements - 1 - i];
  }
}

void reverse_scalar_unroll16(const float* input, float* output, const int* permute, int num_elements) {
  int vec_end = num_elements & (~(VecWidth-1));
  for (int i = 0; i < vec_end; i+= VecWidth) {
    float* dst = &output[i];
    const float* src = &input[num_elements - 1 - i];
    // NOTE(boulos): This version is much faster than doing array
    // indexing while above they're similar with arrays being faster.
    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;

    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;

    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;

    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;
    *dst++ = *src--;
  }
  for (int i = vec_end; i < num_elements; i++) {
    output[i] = input[num_elements - 1 - i];
  }
}


void gather_vector(const float* input, float* output, const int* permute, int num_elements) {
  int vec_end = num_elements & (~(VecWidth-1));
  for (int i = 0; i < vec_end; i+=VecWidth) {
    FixedVector<int, VecWidth> gather_indices(&permute[i], true);
    FixedVector<float, VecWidth> gathered;
    gathered.gather(input, gather_indices, sizeof(float));
    gathered.store_aligned(&(output[i]));
  }
  if (vec_end != num_elements) {
    int i = vec_end;
    FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end);
    FixedVector<int, VecWidth> gather_indices(&permute[i], true);
    FixedVector<float, VecWidth> gathered;
    gathered.gather(input, gather_indices, sizeof(float), store_mask);
    gathered.store_aligned(&(output[i]), store_mask);
  }
}

void copy_vector(const float* input, float* output, const int* permute, int num_elements) {
  int vec_end = num_elements & (~(VecWidth-1));
  for (int i = 0; i < vec_end; i+=VecWidth) {
    FixedVector<float, VecWidth> gathered(&input[i], true);
    gathered.store_aligned(&(output[i]));
  }
  if (vec_end != num_elements) {
    int i = vec_end;
    FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end);
    FixedVector<float, VecWidth> gathered(&input[i], true);
    gathered.store_aligned(&(output[i]), store_mask);
  }
}

void reverse_vector(const float* input, float* output, const int* permute, int num_elements) {
  int vec_end = num_elements & (~(VecWidth-1));
  for (int i = 0; i < vec_end; i+=VecWidth) {
    // Load the vecwidth stuff, and then do it backwards
    FixedVector<float, VecWidth> gathered(&input[num_elements - i - VecWidth], true);
    gathered = reverse(gathered);
    gathered.store_aligned(&(output[i]));
  }
  if (vec_end != num_elements) {
    int i = vec_end;
    FixedVectorMask<VecWidth> store_mask = FixedVectorMask<VecWidth>::FirstN(num_elements - vec_end);
    FixedVector<float, VecWidth> gathered(&input[num_elements - i - VecWidth], true);
    gathered = reverse(gathered);
    gathered.store_aligned(&(output[i]), store_mask);
  }
}

typedef void(*MemCpyFunc)(const float*, float*, const int*, int);

void BenchFunctions(const char* func1_name, MemCpyFunc func1,
                    const char* func2_name, MemCpyFunc func2,
                    const float* input,
                    float* func1_output, float* func2_output,
                    const int* permute, int num_elements,
                    int bench, int warmup) {
  for (int i = 0; i < warmup; i++) {
    func1(input, func1_output, permute, num_elements);
  }
  CycleTimer::SysClock start_func1 = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    func1(input, func1_output, permute, num_elements);
  }
  CycleTimer::SysClock end_func1 = CycleTimer::currentTicks();

  CycleTimer::SysClock func1_ticks = end_func1 - start_func1;
  double func1_time_avg = func1_ticks * CycleTimer::secondsPerTick() / bench;
  CycleTimer::SysClock func1_ticks_per_element = func1_ticks / (num_elements * bench);
  printf("%25s time: %lf (%4.2f million elements per second, %d %s per element)\n", func1_name, func1_time_avg, (num_elements / func1_time_avg) / 1e6f, int(func1_ticks_per_element), CycleTimer::tickUnits());

  if (func2) {
    for (int i = 0; i < warmup; i++) {
      func2(input, func2_output, permute, num_elements);
    }
    CycleTimer::SysClock start_func2 = CycleTimer::currentTicks();
    for (int i = 0; i < bench; i++) {
      func2(input, func2_output, permute, num_elements);
    }
    CycleTimer::SysClock end_func2 = CycleTimer::currentTicks();
    CycleTimer::SysClock func2_ticks = end_func2 - start_func2;
    double func2_time_avg = func2_ticks * CycleTimer::secondsPerTick() / bench;
    CycleTimer::SysClock func2_ticks_per_element = func2_ticks / (num_elements * bench);
    printf("%25s time: %lf (%4.2f million elements per second, %d %s per element)\n", func2_name, func2_time_avg, (num_elements / func2_time_avg) / 1e6f, int(func2_ticks_per_element), CycleTimer::tickUnits());
    printf("%25s vs %s speedup: %lf\n\n", func1_name, func2_name, func1_time_avg / func2_time_avg);

    for (int i = 0; i < num_elements; i++) {
      if (func1_output[i] != func2_output[i]) {
        printf("ERROR: Outputs differ %4d, %s -> %8f != %s -> %8f (input = %8f)\n", i, func1_name, func1_output[i], func2_name, func2_output[i], input[i]);
      }
    }
  }
}

int main(int argc, char** argv) {
  int num_elements = kDefaultNumElements;
  int bench = kNumBench;
  int warmup = kNumWarmup;

  if (argc > 1) num_elements = atoi(argv[1]);

  float* scalar_output = new float[num_elements];
  float* vector_output = new float[num_elements];
  float* input = randomFloats(num_elements, 0xDEADBABE);
  int* permutation = new int[num_elements];
  // Identity for now.
  for (int i = 0; i < num_elements; i++) {
    permutation[i] = i;
  }

#if 1
  BenchFunctions("gather_scalar", gather_scalar,
                 "copy_scalar", copy_scalar,
                 input, scalar_output, vector_output, permutation, num_elements, bench, warmup);

  BenchFunctions("copy_scalar", copy_scalar,
                 "copy_scalar_unroll16", copy_scalar_unroll16,
                 input, scalar_output, vector_output, permutation, num_elements, bench, warmup);

  BenchFunctions("copy_scalar", copy_scalar,
                 "memcpy_scalar", memcpy_scalar,
                 input, scalar_output, vector_output, permutation, num_elements, bench, warmup);

  BenchFunctions("reverse_scalar", reverse_scalar,
                 "reverse_scalar_unroll16", reverse_scalar_unroll16,
                 input, scalar_output, vector_output, permutation, num_elements, bench, warmup);


  BenchFunctions("copy_scalar_unroll16", copy_scalar_unroll16,
                 "copy_vector", copy_vector,
                 input, scalar_output, vector_output, permutation, num_elements, bench, warmup);
#endif
  BenchFunctions("reverse_scalar_unroll16", reverse_scalar_unroll16,
                 "reverse_vector", reverse_vector,
                 input, scalar_output, vector_output, permutation, num_elements, bench, warmup);
  return 0;
}
