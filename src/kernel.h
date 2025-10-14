#pragma once
#include "link.h"
#include <cuda_runtime.h>

bool is_CUDA_available();
void test_random(int N);
void __global__ print_array(int arr[]);
void __global__ print_array(float arr[]);

void __global__ process_node(int * counts, int * thresholds, int * max_flows, double ** cumulative_rates, int ** adjacency_list, double * choices);

void cuda_main();
