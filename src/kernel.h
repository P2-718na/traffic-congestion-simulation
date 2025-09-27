#pragma once
#include <cuda_runtime.h>
#include <vector>

struct Link {
    int to;
    float weight;
};

bool is_CUDA_available();

void __global__ move_agent(int positions[], Link* cumulative_rates[], int thresholds[], int counts[], bool& are_there_failures);

void test_random(int N);
void __global__ print_array(float choices[]);
