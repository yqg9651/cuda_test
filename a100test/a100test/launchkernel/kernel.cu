#include <stdio.h>

extern "C" __global__ void TestKernel(void)
{
    int a = 0;

    printf(" a = %d \n", a);
    while (1)
        a++;
}
