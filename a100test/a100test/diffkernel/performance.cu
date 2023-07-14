#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "math.h"
#include "cuda.h"

#define CHECK(func) \
{   \
    if (func != CUDA_SUCCESS) { \
        printf("Function Call %s Failed!\n", #func);    \
        exit(1);    \
    }   \
}

/********************************/
float Avg(float *Array, int Length)
{
    float sum = 0;
    for (int i = 0; i < Length; i++) {
        sum += Array[i];
    }

    return sum/Length;
}

float Std(float *Array, int Length)
{
    double var, avg;
    avg = Avg(Array, Length);
    for (int i = 0; i < Length; i++) {
        var += pow(Array[i] - avg, 2)/Length;
    }
    return pow(var, 0.5);
}

float Max(float *Array, int Length)
{
    float max = 0;
    for (int i = 0; i < Length; i++) {
        if (max < Array[i])
            max = Array[i];
    }
    return max;
}

float Min(float *Array, int Length)
{
    float min = 999999999999.0;
    for (int i = 0; i < Length; i++) {
        if (min > Array[i])
            min = Array[i];
    }
    return min;
}

/********************************/
CUdevice device;
CUcontext context;
CUmodule module[1000];
CUfunction fast[1000], slow, calcu;
float *ResArray;

void initCUDA(void)
{
    int devCount = 0;

    CHECK(cuInit(0));
    CHECK(cuDeviceGetCount(&devCount));

    if (devCount == 0) {
        printf("No Device!\n");
        exit(1);
    }

    CHECK(cuDeviceGet(&device, 0));
    CHECK(cuCtxCreate(&context, 0, device));

    if (cuCtxResetPersistingL2Cache()) {
        printf("Use Cuda ToolKit Version 10.x\n");
    } else {
        printf("Use Cuda ToolKit Version 11.x\n");
    }

for (int i = 0; i < 1000; i++) {
    CHECK(cuModuleLoad(&module[i], "test.ptx"));
    CHECK(cuModuleGetFunction(&fast[i], module[i], "FastKernel"));
}
    CHECK(cuModuleGetFunction(&slow, module[0], "SlowKernel"));
    CHECK(cuModuleGetFunction(&calcu, module[0], "CalcuKernel"));
}

void exitCUDA(void)
{

for (int i = 0; i < 1000; i++) {
    CHECK(cuModuleUnload(module[i]));
}
    CHECK(cuCtxDestroy(context));
}

void PipeLineTime(int cycle_num)
{
    CUstream stream;
    CUdeviceptr d_a;
    struct timeval start, end;
    float sw_time = 0;
    void *params[] = {&d_a};

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CHECK(cuMemAlloc(&d_a, 1024));

    gettimeofday(&start, NULL);
    for (int i = 0; i < cycle_num; i++) {
            CHECK(cuLaunchKernel(fast[0],
                1, 1, 1,
                1, 1, 1,
                0, stream, params, NULL));
    }

    CHECK(cuStreamSynchronize(stream));
    gettimeofday(&end, NULL);

    sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
        (start.tv_sec * 1000000 + start.tv_usec);

    printf("total time = %f us, single time = %f us\n", sw_time, sw_time/cycle_num);

    gettimeofday(&start, NULL);
    for (int i = 0; i < cycle_num; i++) {
            CHECK(cuLaunchKernel(fast[cycle_num % 1000],
                1, 1, 1,
                1, 1, 1,
                0, stream, params, NULL));
    }

    CHECK(cuStreamSynchronize(stream));
    gettimeofday(&end, NULL);

    sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
        (start.tv_sec * 1000000 + start.tv_usec);

    printf("diff kernel total time = %f us, single time = %f us\n", sw_time, sw_time/cycle_num);
}

int main(int argc, char **argv)
{
    ResArray = (float *)calloc(8192, sizeof(float));
    if (!ResArray) {
        printf("Malloc Failed!\n");
        return 1;
    }

    printf("- Init...\n");
    initCUDA();

    PipeLineTime(8192);
    //LaunchDelay(8192);

    exitCUDA();
    return 0;
}
