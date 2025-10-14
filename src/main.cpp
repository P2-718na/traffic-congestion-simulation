#include "kernel.h"

int main()
{
    is_CUDA_available();
    //test_random(10000);
    cuda_main();
}
