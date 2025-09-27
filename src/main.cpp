#include "kernel.h"

int main()
{
    int N = 4; // agent count
    int M = 100; // node count
    int t = 20; // Max simulation length

    is_CUDA_available();
    test_random(100000);

    //NaiveSimulation<10> sim{};
}
