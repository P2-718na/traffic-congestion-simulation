#include "kernel.h"

#include <cstdio>
#include <curand.h>

void check_CUDA() {
    int count{};
    const auto error = cudaGetDeviceCount(&count);

    std::printf("Found %i GPUs\n", count);
    if (error != 0) {
        exit(1);
    }
}

// Move a single agent.
// Agent position is stored into positions array, with index the same as
// block id
template<int _NM>
std::vector<int> NaiveSimulation<_NM>::simulate() {
 for (int t = 0; t < tMax; ++t) {
  move_agents();

  if (are_there_failures) {
   break;
  }
 }

 if (!are_there_failures) {
  return {};
 }

 __global__ std::vector<int>avalanche_sizes{congested_nodes};
 map_one_avalanche_size<<<1, avalanche_sizes.size()>>>(avalanche_sizes);
 return avalanche_sizes;
}

template<int _NM>
void NaiveSimulation<_NM>::move_agents() {
 auto __global__ choices = new float[N];
 curandGenerateUniform(generator, choices, N);
 move_agent<<<1, N>>>(choices);
 // TODO check that this delete happens after all threads have finished
 delete[] choices;
}

template<int _NM>
void NaiveSimulation<_NM>::move_agent(float choices[]) {
 const int idx = blockIdx.x;

 std::printf("%f\n", choices[idx]);

 /*RandomChoice[rateMatrix[[;; , #]] -> possibleAgentPositions ]
         ) & /@ agentPositions;*/
 //TODO here, for now try to get random numbers to be printed
}

template<int _NM>
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
     ];*/
 return 1;
}
