#include <stdio.h>
#include <stdint.h>

extern "C" __global__ void FastKernel(int *a)
{
}

extern "C" __global__ void SlowKernel(int *a)
{
    int b = *a;
    for (int i = 0; i < (b%10)+20000; i++) {
        *a = i + *a;
    }
}

#define SIZE    (0x2UL << 20)
extern "C" __global__ void CalcuKernel(int *a, int *b)
{
    for (int i = 0; i < SIZE/4; i++) {
        b[i] = a[i] + i;
    }
}
