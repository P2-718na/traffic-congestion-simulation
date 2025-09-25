#pragma once

#include <array>
#include <curand.h>
#include <vector>

void check_CUDA();

template<int _NM>
class NaiveSimulation {

    static constexpr int M{_NM};

    int N;
    int tMax;

    std::vector<int> congested_nodes{};

    __shared__ std::array<int, _NM>  thresholds; // Kernels need fast access to this to check thresholds atE of each step. so, shared.
    __shared__ std::vector<int> positions{}; // again, need fast access to read and write positions

    std::array< std::array<float, _NM>, _NM > rate_matrix{};
    std::array< std::vector<float>, _NM > precomputed_cumulative_rates{};

    curandGenerator_t generator;

public:
    std::vector<int> simulate();
    void move_agents();

    bool are_there_failures{false};

    void move_agent(float choices[]);
    int map_one_avalanche_size(std::vector<int>& congested_nodes);

    // These guys need to be compiled here otherwise things break!

    NaiveSimulation() {
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, 0ULL);
        // TODO

        /*
       Simulate[rateMatrix_, nodeThresholds_, startingAgentPositions_,
          maxSteps_] :=
         Module[{MoveAgents, AreThereFailures , ComputeAvalancheSize, i,
           nNodes, nAgents, carCountsByNode, agentPositions,
           possibleAgentPositions, failedNodes},
          nNodes = Length[rateMatrix];
            agentPositions = (
                 Assert[Total[rateMatrix[[;; , #]]] == 1];

       */

    }
    ~NaiveSimulation() {
        curandDestroyGenerator(generator);
    }
    NaiveSimulation(NaiveSimulation &) = delete;
};

