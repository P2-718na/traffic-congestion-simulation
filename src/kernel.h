#pragma once

#include <array>
#include <curand.h>
#include <vector>

void check_CUDA();

template<int _NM>
__global__ class NaiveSimulation {
    static constexpr int M{_NM};

private:
    int N;
    int tMax;

    std::vector<int> congested_nodes{};

    __shared__ std::array<int, _NM>  thresholds; // Kernels need fast access to this to check thresholds atE of each step. so, shared.
    __shared__ std::vector<int> positions{}; // again, need fast access to read and write positions

    std::array< std::array<float, _NM>, _NM > rate_matrix{};
    std::array< std::vector<float>, _NM > precomputed_cumulative_rates{};

    curandGenerator_t generator;

public:
    NaiveSimulation();
    ~NaiveSimulation();
    NaiveSimulation(NaiveSimulation &) = delete;

    std::vector<int> simulate();
    void move_agents();

    bool are_there_failures{false};

    void __global__ move_agent(float choices[]);
    int __global__  map_one_avalanche_size(std::vector<int>& congested_nodes);
};

