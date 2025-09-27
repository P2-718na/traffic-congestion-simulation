#include "kernel.h"
#include "simulation.h"

int main()
{
    int N = 1; // agent count
    int M = 3; // node count
    int t = 20; // Max simulation length

    is_CUDA_available();
    //test_random(10000);

    NaiveSimulation sim{M, N, t, {2, 2, 2}, {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}};
    sim.print_positions();
    sim.move_agents();
    sim.print_positions();
    printf("Done!\n");

    test_random(10);
}
