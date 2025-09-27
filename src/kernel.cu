#include "kernel.h"
#include "link.h"
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

void __global__ move_agent(int* positions, Link** cumulative_rates, float choices[], int thresholds[], int counts[], bool* are_there_failures) {
  const auto idx = blockIdx.x;

  const auto old_position = positions[idx];
  const auto choice = choices[idx];
  auto rates = cumulative_rates[idx];

  // Weights go from 0 to 1 inclusive, random numbers are 0 exclusive to 1 inclusive.
  // In theory there should be no risk of overflow, if data is properly formatted.
  while ( (rates++)->weight < choice )
  {}
  const auto new_position = rates->to; // For some reason -> works only on bare pointer types

  --counts[old_position];
  ++counts[new_position];

  if (thresholds[new_position] > counts[new_position])
  {
    *are_there_failures = true;
    // Todo check that this is not already present!!
  }
}

void __global__ randomize_positions(int M, int* positions, float choices[])
{
  auto idx = blockIdx.x;
  positions[idx] = (int)choices[idx]*M;
}
void __global__ compute_counts(int* positions, int counts[])
{
  auto idx = blockIdx.x;
  const auto position = positions[idx];
  ++counts[position];
}
void __global__ compute_cumulative_rates(int M, float** rate_matrix, Link** precomputed_cumulative_rates)
{
  const auto idx = blockIdx.x;

  for (int from = 0; from < M; ++from)
  {
    constexpr int epsilon = 1e-5;
    float cum_rates{0};
    int link_count{0};
    for (int to = 0; to < M; ++to)
    {
      const auto next_rate = rate_matrix[from][to];
      if (next_rate > epsilon)
      {
        cum_rates += next_rate;
        precomputed_cumulative_rates[idx][link_count++] = Link{to, cum_rates}; //TODO
      }
    }
    if (link_count <= 1) //TODO
    {
      // TODO handle this
      printf("'From' node: %i is a sink!\n", from);
    }
  }
}


// TEST /////
void __global__ print_array(int arr[]) {
  const auto idx = blockIdx.x;
  printf("%i -> %i\n", idx, arr[idx]);
}
void __global__ print_array(float arr[]) {
  const auto idx = blockIdx.x;
  printf("%i -> %f\n", idx, arr[idx]);
}

void test_random(int N) {
  curandGenerator_t generator;
  float* choices;

  cudaMalloc(&choices, N*sizeof(float));
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(generator, 0ULL);

  curandGenerateUniform(generator, choices, N);
  print_array<<<N, 1, 1>>>(choices);
  cudaDeviceSynchronize();

  cudaFree(choices);
  curandDestroyGenerator(generator);
}
