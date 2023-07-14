#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "cuda.h"

#define CHECK(func) \
{   \
    if (func != CUDA_SUCCESS) { \
        printf("Function Call %s Failed!\n", #func);    \
        exit(1);    \
    }   \
}

#define THREAD_NUM  (1)

CUdevice device;
CUcontext context;
CUmodule module;
CUfunction function;
pthread_mutex_t lock;
uint64_t count = 0;

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

    CHECK(cuModuleLoad(&module, "kernel.ptx"));
    CHECK(cuModuleGetFunction(&function, module, "TestKernel"));
}

void __HostFunc(void *data)
{
    // just loop
    while (1) {
        sleep(1);
    }
}

uint32_t tid = 0;
void *ThreadFunc(void *data)
{
    CUstream stream;
    CUdeviceptr d_a, d_b;
    void *h_a;
    uint32_t ltid = __sync_fetch_and_add(&tid, 1);
    uint32_t lcount = 0;

    CHECK(cuCtxSetCurrent(context));
    for (;;) {
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        printf("stream number is %d\n", lcount++);
    }

    return NULL;
    CHECK(cuMemAlloc(&d_a, 1024));
    CHECK(cuMemAlloc(&d_b, 1024));
    CHECK(cuMemAllocHost((void **)&h_a, 1024));

//    CHECK(cuLaunchHostFunc(stream, __HostFunc, NULL));
    for (;;) {
        CHECK(cuLaunchKernel(function,
            1, 1, 1,
            1, 1, 1,
            0, stream, NULL, NULL));

//        CHECK(cuLaunchHostFunc(stream, __HostFunc, NULL));
        CHECK(cuMemcpyHtoDAsync(d_a, h_a, 1024, stream));

        pthread_mutex_lock(&lock);
        printf("[%d] count = %ld, local count = %d\n",
                ltid, __sync_add_and_fetch(&count, 1),
                lcount++);
        pthread_mutex_unlock(&lock);
    }

    return NULL;
}

int main(int argc, char **argv)
{
    pthread_t _id[THREAD_NUM] = {0};

    printf("- Init...\n");
    initCUDA();

    pthread_mutex_init(&lock, 0);

    for (int i = 0; i < THREAD_NUM; i++) {
        CHECK(pthread_create(&_id[i], NULL, ThreadFunc, NULL));
    }

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(_id[i], NULL);
    }

//    CHECK(cuStreamSynchronize(0));

    return 0;
}
