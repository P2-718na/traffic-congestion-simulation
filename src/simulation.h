#pragma once

#include "link.h"
#include "kernel.h"
#include <curand.h>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>

using namespace std;

void check_CUDA();

class NaiveSimulation {

  int M; // number of nodes
  int N; // number of agents
  int tMax; // max simulation time

  // Dynamic size device stuff DOES NOT EXISTS. IT IS NOT REAL. YOU WILL BE HURT.
  // // cannot use  thrust::device_vector<int>, no push_back in device code
  int* positions;
  int* asd;

  Link*/***/ precomputed_cumulative_rates; // M*(variable)
  float*/***/ rate_matrix; // M*M

  // Static-size device stuff. Although not statically allocated. lol.
  float* choices;
  int* thresholds; // M
  int* counts; // M
  bool* are_there_failures; // 1

  // Host vars
  curandGenerator_t generator;

public: //todo this is here just for testing
  inline void move_agents() {
    curandGenerateUniform(generator, choices, N);
    move_agent<<<N, 1, 1>>>(M, positions, precomputed_cumulative_rates, choices, thresholds, counts, are_there_failures);
    cudaDeviceSynchronize();
  }

public:
  inline NaiveSimulation(int M, int N, int tMax, vector<int> thresholds, vector< vector<float> > rate_matrix) : M{M}, N{N}, tMax{tMax} {
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 0ULL);
    cudaDeviceSynchronize();

    cudaMalloc(&(this->positions), N*sizeof(int));
    cudaMalloc(&(this->precomputed_cumulative_rates), M*M*sizeof(Link));
    cudaMalloc(&choices, N*sizeof(float));
    cudaMalloc(&(this->thresholds), M*sizeof(int));
    cudaMalloc(&(this->counts), M*sizeof(int));
    cudaMalloc(&(this->rate_matrix), M*M*sizeof(float));
    cudaMalloc(&(this->are_there_failures), sizeof(bool));

    cudaMemcpy(this->thresholds, &thresholds.front(), M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(this->counts, 0, M*sizeof(int));
    cudaMemset(are_there_failures, 0, sizeof(bool));
    for (int from = 0; from < M; ++from)
    {
      cudaMemcpy(this->rate_matrix + (from*M), &rate_matrix[from].front(), M*sizeof(float), cudaMemcpyHostToDevice);
    }
    curandGenerateUniform(generator, choices, N);
    cudaDeviceSynchronize(); // For some reason memset is async lol diocane

    randomize_positions<<<N, 1, 1>>>(M, positions, choices);
    cudaDeviceSynchronize(); // Counts need positions to be set

    compute_counts<<<N, 1, 1>>>(positions, counts);
    compute_cumulative_rates<<<M, 1, 1>>>(M, this->rate_matrix, precomputed_cumulative_rates);
    cudaDeviceSynchronize();
  }
  ~NaiveSimulation() {
    //curandDestroyGenerator(generator);
  }
  NaiveSimulation(NaiveSimulation &) = delete;

  inline std::vector<int> simulate(bool print = false) {
    bool has_failed;

    for (int t = 0; t < tMax; ++t) {
      move_agents();

      cudaMemcpy(&has_failed, are_there_failures, sizeof(bool), cudaMemcpyDeviceToHost);

      if (print)
      {
        print_positions();
      }

      if (has_failed) {
        printf("NO %i\n", has_failed);
          break;
      }
    }

    if (!has_failed) {
      return {};
    }

    // todo copy to vector
    // computeAvalanches...
    return {};
  }

  inline void print_positions()
  {
    printf("Positions:\n");
    print_array<<<N, 1, 1>>>(positions);
    cudaDeviceSynchronize();
    printf("End of list\n");
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
