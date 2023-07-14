#include <stdio.h>

extern "C" __global__ void TestKernel(int size, int *data1)
{
	for (int i = 0; i < size/4 - 1; i++) {
		data1[i] = data1[i+1]+1;
	}
}
