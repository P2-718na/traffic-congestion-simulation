#include "kernel.h"
#include <curand.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <thrust/reduce.h>

bool is_CUDA_available() {
  int count{};
  const auto error = cudaGetDeviceCount(&count);

  return error == 0;
}

void __global__ move_agent(int positions[], Link* cumulative_rates[], float choices[], int thresholds[], int counts[], bool& are_there_failures) {
  const auto idx = blockIdx.x;

  const auto old_position = positions[idx];
  const auto choice = choices[idx];
  auto rates = cumulative_rates[idx];

  // Weights go from 0 to 1 inclusive, random numbers are 0 exclusive to 1 inclusive.
  // In theory there should be no risk of overflow, if data is properly formatted.
  while ( (rates++)->weight < choice )
  {}
  const auto new_position = rates->to;

  --counts[old_position];
  ++counts[new_position];

  if (thresholds[new_position] > counts[new_position])
  {
    are_there_failures = true;
  }
}

void __global__ print_array(float choices[]) {
  const auto idx = blockIdx.x;
  printf("%d -> %f\n", idx, choices[idx]);
}

void test_random(int N) {
  curandGenerator_t generator;
  float* choices;

  cudaMalloc(&choices, N*sizeof(float));
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(generator, 0ULL);

  curandGenerateUniform(generator, choices, N);
  print_array<<<N, 1, 1>>>(choices);

  cudaFree(choices);
  curandDestroyGenerator(generator);
}
