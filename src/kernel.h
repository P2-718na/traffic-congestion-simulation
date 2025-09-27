#pragma once

#include "link.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>


bool is_CUDA_available();


void __global__ move_agent(int* positions, Link** cumulative_rates, float choices[], int thresholds[], int counts[], bool* are_there_failures);
void __global__ randomize_positions(int M, int* positions, float choices[]);
void __global__ compute_counts(int* positions, int counts[]);
void __global__ compute_cumulative_rates(int M, float** rate_matrix, Link** cumulative_rates);

void test_random(int N);
void __global__ print_array(int arr[]);
void __global__ print_array(float arr[]);
