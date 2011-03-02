#include "syrah/FixedVector.h"
#include "syrah/CycleTimer.h"
using namespace syrah;
static const int VecWidth = 16;
static const int kDefaultNumElements = 1024;
static const int kNumWarmup = 16;
static const int kNumBench  = 1024;

float* randomFloats(int how_many, unsigned int seed) {
  float* result = new float[how_many];
  unsigned int lcg = seed;
  for (int i = 0; i < how_many; i++) {
    lcg = 1664525 * lcg + 1013904223;
    result[i] = -10.f + 20.f * (lcg / 4294967296.f);
  }
  return result;
}

#if 0
void gather_scalar(float* input, float* output, int* permute, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    output[i] = input[permute[i]];
  }
}
#else
void gather_scalar(float* input, float* output, int* permute, int num_elements) {
#if 0
  for (int i = 0; i < num_elements; i++) {
    output[i] = input[i];
  }
#else
#if 0
  int vec_end = num_elements & (~(VecWidth-1));
  for (int i = 0; i < vec_end; i+= VecWidth) {
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
  }
  if (vec_end != num_elements) {
    for (int i = vec_end; i < num_elements; i++) {
      output[i] = input[i];
    }
  }
#else
  memcpy(output, input, sizeof(float) * num_elements);
#endif
#endif
}
#endif

#if 0
void gather_vector(float* input, float* output, int* permute, int num_elements) {
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
#else
void gather_vector(float* input, float* output, int* permute, int num_elements) {
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
#endif

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

  for (int i = 0; i < warmup; i++) {
    gather_scalar(input, scalar_output, permutation, num_elements);
  }
  CycleTimer::SysClock start_scalar = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    gather_scalar(input, scalar_output, permutation, num_elements);
  }
  CycleTimer::SysClock end_scalar = CycleTimer::currentTicks();

  for (int i = 0; i < warmup; i++) {
    gather_vector(input, vector_output, permutation, num_elements);
  }
  CycleTimer::SysClock start_vector = CycleTimer::currentTicks();
  for (int i = 0; i < bench; i++) {
    gather_vector(input, vector_output, permutation, num_elements);
  }
  CycleTimer::SysClock end_vector = CycleTimer::currentTicks();

  CycleTimer::SysClock scalar_ticks = end_scalar - start_scalar;
  CycleTimer::SysClock vector_ticks = end_vector - start_vector;

  double scalar_time_avg = scalar_ticks * CycleTimer::secondsPerTick() / bench;
  double vector_time_avg = vector_ticks * CycleTimer::secondsPerTick() / bench;

  CycleTimer::SysClock scalar_ticks_per_element = scalar_ticks / (num_elements * bench);
  CycleTimer::SysClock vector_ticks_per_element = vector_ticks / (num_elements * bench);

  printf("%s scalar time: %lf (%4.2f million elements per second, %d %s per element)\n", "gather", scalar_time_avg, (num_elements / scalar_time_avg) / 1e6f, int(scalar_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector time: %lf (%4.2f million elements per second, %d %s per element)\n", "gather", vector_time_avg, (num_elements / vector_time_avg) / 1e6f, int(vector_ticks_per_element), CycleTimer::tickUnits());
  printf("%s vector speedup: %.2f\n", "memcpy", scalar_time_avg/vector_time_avg);

  return 0;
}
