#pragma once

#include "link.h"
#include "kernel.h"
#include <curand.h>
#include <vector>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

using namespace std;

void check_CUDA();

class NaiveSimulation {

  int M; // number of nodes
  int N; // number of agents
  int tMax; // max simulation time

  // Dynamic size device stuff DOES NOT EXISTS. IT IS NOT REAL. YOU WILL BE HURT.
  // // cannot use  thrust::device_vector<int>, no push_back in device code
  int* positions;
  Link** precomputed_cumulative_rates; // M*(variable)

  // Static-size device stuff. Although not statically allocated. lol.
  float* choices;
  int* thresholds{}; // M
  int* counts{}; // M
  float** rate_matrix{}; // M*M
  bool* are_there_failures{}; // 1

  // Host vars
  curandGenerator_t generator;

public: //todo this is here just for testing
  inline void move_agents() {
    curandGenerateUniform(generator, choices, N);
    move_agent<<<N, 1, 1>>>(positions, precomputed_cumulative_rates, choices, thresholds, counts, are_there_failures);
    cudaDeviceSynchronize();
  }

public:
  inline NaiveSimulation(int M, int N, int tMax, vector<int> thresholds, vector< vector<float> > rate_matrix) : M{M}, N{N}, tMax{tMax} {
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 0ULL);

    cudaMalloc(&(this->positions), M*sizeof(int));
    cudaMalloc(&(this->precomputed_cumulative_rates), M*sizeof(thrust::device_vector<Link>));
    cudaMalloc(&choices, N*sizeof(float));
    cudaMalloc(&(this->thresholds), M*sizeof(int));
    cudaMalloc(&(this->counts), M*sizeof(int));
    cudaMalloc(&(this->rate_matrix), M*M*sizeof(int));
    cudaMalloc(&(this->are_there_failures), sizeof(bool));
    cudaMemcpy(this->thresholds, &thresholds.front(), M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(this->counts, 0, M*sizeof(int));
    cudaMemset(are_there_failures, 0, sizeof(bool));
    for (int i = 0; i < M; ++i)
    {
     cudaMemcpy(this->rate_matrix + i, &rate_matrix[i].front(), M*sizeof(float), cudaMemcpyHostToDevice);
    }

    curandGenerateUniform(generator, choices, N);
    randomize_positions<<<N, 1, 1>>>(M, positions, choices);
    cudaDeviceSynchronize();

    compute_counts<<<N, 1, 1>>>(positions, counts);
    compute_cumulative_rates<<<M, 1, 1>>>(M, this->rate_matrix, precomputed_cumulative_rates);

    cudaDeviceSynchronize();
  }
  ~NaiveSimulation() {
    curandDestroyGenerator(generator);

    cudaFree(positions);
    cudaFree(precomputed_cumulative_rates);
    cudaFree(choices);
    cudaFree(thresholds);
    cudaFree(counts);
    cudaFree(rate_matrix);
    cudaFree(are_there_failures);
  }
  NaiveSimulation(NaiveSimulation &) = delete;

  inline std::vector<int> simulate() {
    for (int t = 0; t < tMax; ++t) {
      move_agents();

      if (are_there_failures) {
          break;
      }
    }

    if (!are_there_failures) {
      return {};
    }

    // todo copy to vector
    // computeAvalanches...
    return {};
  }

  inline void print_positions()
  {
      int* counts = (int*)malloc(M*sizeof(int));
      cudaMemcpy(counts, this->counts, M*sizeof(int), cudaMemcpyDeviceToHost);
      for (int i = 0; i < M; ++i)
      {
          printf("%i -> %i\n", i, counts[i]);
      }
  }
};



/*
int NaiveSimulation<_NM>::map_one_avalanche_size(std::vector<int>& congested_nodes) {
 // TODO

 /*
   ComputeAvalancheSize[node_] :=
    Module[{avalancheSize, carCount, avalancheCarCounts,
      adjacentNodes, nodesToCheck, lockedNodes, currentNode, newNode,
      transitionProbabilities},
     nodesToCheck = {node};
     lockedNodes = {};
     avalancheCarCounts = carCountsByNode;


     While[Length@nodesToCheck > 0,
      currentNode = Pop[nodesToCheck];
      Assert[avalancheCarCounts[[currentNode]] > 0, "carCount"];

      If[
       nodeThresholds[[currentNode]] -
         avalancheCarCounts[[currentNode]] >= 0, Continue[]];

      AppendTo[lockedNodes, currentNode];

      If[Length@lockedNodes == nNodes, Break[]];

      transitionProbabilities = Normal@rateMatrix[[currentNode]];
      (transitionProbabilities[[#]] = 0 ) & /@ lockedNodes;

      (* If there is nowhere to go,
      we just lock this node and no transfer*)
      If[Total@transitionProbabilities == 0,
       Continue[];
       ];

      transitionProbabilities /= Total[transitionProbabilities];

      For[carCount = avalancheCarCounts[[currentNode]], carCount > 0,
       carCount--,
       Assert[nNodes == Length@transitionProbabilities, "nNodes"];

       newNode =
        RandomChoice[
         transitionProbabilities -> Table[x, {x, 1, nNodes}]];

       Assert[newNode != currentNode, "currentNode"];
       Assert[! MemberQ[lockedNodes, newNode], "lockedNodes"];
       If[! MemberQ[nodesToCheck, newNode],
        AppendTo[nodesToCheck, newNode]];

       avalancheCarCounts[[newNode]]++;
       avalancheCarCounts[[currentNode]]--;
       ];
      ];

     Assert[Total[avalancheCarCounts] == Total[carCountsByNode]];

     Return[{node, Length@lockedNodes}];
     ];
 return 1;
}
*/
