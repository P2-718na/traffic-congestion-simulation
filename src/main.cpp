#include <vector>

#include "kernel.h"

int main()
{
    int N = 4; // agent count
    int M = 100; // node count
    int t = 20; // Max simulation length


    check_CUDA();
    NaiveSimulation<10> sim{};
}