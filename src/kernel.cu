#include "kernel.h"
#include "link.h"
#include <curand.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include "simulation.h"

bool is_CUDA_available() {
  int count{};
  const auto error = cudaGetDeviceCount(&count);

  return error == 0;
}

void __global__ move_agent(int M, int* positions, Link* cumulative_rates, float choices[], int thresholds[], int counts[], bool* are_there_failures) {
  const auto idx = blockIdx.x;

  const auto& old_position = positions[idx];
  const auto choice = choices[idx];
  auto rates = cumulative_rates + old_position*M;

  // Weights go from 0 to 1 inclusive, random numbers are 0 exclusive to 1 inclusive.
  // In theory there should be no risk of overflow, IF data is properly formatted.
  while ( rates->weight < choice )
  {
    ++rates;
  }

  const auto new_position = rates->to; // For some reason -> works only on bare pointer types
  positions[idx] = new_position;
  --counts[old_position];
  ++counts[new_position];

  if (thresholds[new_position] > counts[new_position])
  {
    *are_there_failures = true;
  }
}

void __global__ randomize_positions(int M, int* positions, float choices[])
{
  auto idx = blockIdx.x;
  positions[idx] = choices[idx]*(float)M;
}
void __global__ compute_counts(int* positions, int* counts)
{
  auto idx = blockIdx.x;
  const auto position = positions[idx];
  ++counts[position];
}
void __global__ compute_cumulative_rates(int M, float* rate_matrix, Link* precomputed_cumulative_rates)
{
  const auto from = blockIdx.x;

  constexpr int epsilon = 1e-5;
  float cum_rates{0};
  int link_count{0};
  for (int to = 0; to < M; ++to)
  {
    const auto next_rate = rate_matrix[from * M + to];
    if (next_rate > epsilon)
    {
      cum_rates += next_rate;
      precomputed_cumulative_rates[from * M + link_count++] = Link{to, cum_rates}; //TODO
      printf("FROM: %i, TO: %i, w: %f\n", from, to, cum_rates);
    }
  }
  if (link_count <= 1) //TODO
  {
    // TODO handle this
    printf("'From' node: %i might be a sink!\n", from);
  }
  auto& last_item = precomputed_cumulative_rates[from * M + link_count++].to;
  if (last_item < 1)
  {
    // Avoid rounding errors
    last_item = 1;
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
  cudaDeviceSynchronize();

  curandGenerateUniform(generator, choices, N);
  print_array<<<N, 1, 1>>>(choices);
  cudaDeviceSynchronize();

  cudaFree(choices);
  curandDestroyGenerator(generator);
}

void test_sim()
{
  int N = 1; // agent count
  int M = 3; // node count
  int t = 20; // Max simulation length

  NaiveSimulation sim{M, N, t, {2, 2, 2}, {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}};
  sim.print_positions();
  sim.move_agents();
  sim.print_positions();
  sim.move_agents();
  sim.print_positions();
  printf("Done!\n");
}
