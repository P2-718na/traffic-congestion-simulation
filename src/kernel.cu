#include "kernel.h"

#include "rapidcsv.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdio>

typedef unsigned long long ull;

bool is_CUDA_available() {
  int count{};
  const auto error = cudaGetDeviceCount(&count);

  return error == 0;
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


void __global__ print_array(int arr[]) {
  const auto idx = blockIdx.x;
  printf("%i -> %i\n", idx, arr[idx]);
}
void __global__ print_array(float arr[]) {
  const auto idx = blockIdx.x;
  printf("%i -> %f\n", idx, arr[idx]);
}

// ALGORITHMS //////////
void __global__ process_node(double * cumRates, long * linkedList, long totalNodes, long maxLinks, uint * seeds, long * flows, double * thresholds, long * counts)
{
  long nodeId = blockIdx.x;
  printf("%d\n", nodeId);
  long carsToMove = flows[nodeId];
  printf("1\n");
  long movedCars = 0;
  long carCount = counts[nodeId];

  curandStateMRG32k3a_t rand_state;
  curand_init(seeds[nodeId], 0ull, 0ull, &rand_state);

  // Load row into memory
  //__shared__ char * array;
  //auto * choices = (double*)array;
  //const auto choicesOffset =  maxLinks * sizeof(double);
  //auto * links = (long*)array + choicesOffset;
  for (long i = 0; i < maxLinks; ++i)
  {
    printf("%ld, %ld, %f\n", nodeId, linkedList[i + nodeId * totalNodes], cumRates[i + nodeId * totalNodes]);
    //choices[i] = cumRates[i + nodeId * totalNodes]; // Todo compare with docs if this is the better way (column/row first)
    //links[i]   = linkedList[i + nodeId * totalNodes]; // Todo compare with docs if this is the better way (column/row first)
    __syncthreads();
  }

  // Move cars according to flow. If destination is congested, skip move.
  while (carCount > 0 && movedCars < carsToMove)
  {
    // Move a single car
    double choice = curand_uniform(&rand_state); // Choice of current car, in the range ]0, 1]




    movedCars++;
  }
}


// Running
// I guess you cannot spawn kernels from main
// Actually, they have to be spawned from the SAME TRANSLATION UNIT
curandGenerator_t generator;
uint* seeds;

void setup_RNG(long N)
{
  cudaMalloc(&seeds, N*sizeof(uint));
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator, 0ULL);
  cudaDeviceSynchronize();

  curandGenerate(generator, seeds, N);
  cudaDeviceSynchronize();
}

void cuda_main()
{
  long N = 5;
  long max_links = 1;
  __global__ double w[] =  {1, 1, 1, 1, 1};
  __global__ long l[] = {1, 2, 3, 4, 0};
  __global__ long f[] = {5, 5, 5, 5, 5};
  __global__ double c[] = {2, 2, 2, 2, 2};
  __global__ long counts[] = {2, 2, 2, 2, 2};

  setup_RNG(N);

  process_node<<<N, 1>>>(w, l, N, max_links, seeds, f, c, counts);

  cudaDeviceSynchronize();
}
