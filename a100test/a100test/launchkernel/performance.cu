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
CUmodule module;
CUfunction fast, slow, calcu;
float *ResArray;
float *HostTime;
float *DevTime;

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

    //if (cuCtxResetPersistingL2Cache()) {
    //    printf("Use Cuda ToolKit Version 10.x\n");
    //} else {
    //    printf("Use Cuda ToolKit Version 11.x\n");
    //}

    CHECK(cuModuleLoad(&module, "test.ptx"));
    CHECK(cuModuleGetFunction(&fast, module, "FastKernel"));
    CHECK(cuModuleGetFunction(&slow, module, "SlowKernel"));
    CHECK(cuModuleGetFunction(&calcu, module, "CalcuKernel"));
}

void exitCUDA(void)
{
    CHECK(cuModuleUnload(module));
    CHECK(cuCtxDestroy(context));
}

void __HostFunc(void *data)
{
}

float __cal_time(struct timeval *start, struct timeval *end)
{
    return (end->tv_sec * 1000000 + end->tv_usec) -
            (start->tv_sec * 1000000 + start->tv_usec);
}

CUgraph graphStriaghtLine;
CUgraphExec graphSLExec;
void ConstructStriaghtLine(void)
{
#define SIZE    (0x2UL << 20)
    CUdeviceptr d_a, d_b;
    CUgraphNode kNode[32];
    CUDA_KERNEL_NODE_PARAMS kNodeParams = {0};

    CHECK(cuMemAlloc(&d_a, SIZE));
    CHECK(cuMemAlloc(&d_b, SIZE));

    void *params[] = {&d_a, &d_b};
    kNodeParams.blockDimX = 1;
    kNodeParams.blockDimY = 1;
    kNodeParams.blockDimZ = 1;
    kNodeParams.func = fast;
    kNodeParams.gridDimX = 1;
    kNodeParams.gridDimY = 1;
    kNodeParams.gridDimZ = 1;
    kNodeParams.kernelParams = params;

    CHECK(cuGraphCreate(&graphStriaghtLine, 0));

    CHECK(cuGraphAddKernelNode(&kNode[0], graphStriaghtLine,
            NULL, 0, &kNodeParams));
    for (int i = 1; i < 32; i++) {
        CHECK(cuGraphAddKernelNode(&kNode[i], graphStriaghtLine,
                    &kNode[i-1], 1, &kNodeParams));
    }

    CHECK(cuGraphInstantiate(&graphSLExec, graphStriaghtLine, 0));
}

CUgraph graph2;
CUgraphExec graph2Exec;
void ConstructGraph2(void)
{
#define SIZE    (0x2UL << 20)
    CUdeviceptr d_a, d_b;
    int *h_a, *h_b;
    CUgraphNode mNode1, kNode1, kNode2, kNode3, kNode4, mNode2;
    CUDA_KERNEL_NODE_PARAMS kNodeParams = {0};
    CUDA_MEMCPY3D mParams1 = {0};
    CUDA_MEMCPY3D mParams2 = {0};

    CHECK(cuMemAlloc(&d_a, SIZE));
    CHECK(cuMemAlloc(&d_b, SIZE));
    CHECK(cuMemAllocHost((void **)&h_a, SIZE));
    CHECK(cuMemAllocHost((void **)&h_b, SIZE));

    mParams1.Depth = 1;
    mParams1.Height = 1;
    mParams1.WidthInBytes = SIZE;
    mParams1.dstDevice = d_a;
    mParams1.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    mParams1.srcHost = h_a;
    mParams1.srcMemoryType = CU_MEMORYTYPE_HOST;

    mParams2.Depth = 1;
    mParams2.Height = 1;
    mParams2.WidthInBytes = SIZE;
    mParams2.dstHost = h_b;
    mParams2.dstMemoryType = CU_MEMORYTYPE_HOST;
    mParams2.srcDevice = d_b;
    mParams2.srcMemoryType = CU_MEMORYTYPE_DEVICE;

    void *params[] = {&d_a, &d_b};
    kNodeParams.blockDimX = 1;
    kNodeParams.blockDimY = 1;
    kNodeParams.blockDimZ = 1;
    kNodeParams.func = calcu;
    kNodeParams.gridDimX = 1;
    kNodeParams.gridDimY = 1;
    kNodeParams.gridDimZ = 1;
    kNodeParams.kernelParams = params;

    CHECK(cuGraphCreate(&graph2, 0));
    CHECK(cuGraphAddMemcpyNode(&mNode1, graph2,
                NULL, 0, &mParams1, context));
    CHECK(cuGraphAddKernelNode(&kNode1, graph2,
                &mNode1, 1, &kNodeParams));
    CHECK(cuGraphAddKernelNode(&kNode2, graph2,
                &mNode1, 1, &kNodeParams));
    CHECK(cuGraphAddKernelNode(&kNode3, graph2,
                &mNode1, 1, &kNodeParams));
    CHECK(cuGraphAddKernelNode(&kNode4, graph2,
                &mNode1, 1, &kNodeParams));

    CUgraphNode depend[4] = {kNode1, kNode2, kNode3, kNode4};
    CHECK(cuGraphAddMemcpyNode(&mNode2, graph2,
                depend, 4, &mParams2, context));

    CHECK(cuGraphInstantiate(&graph2Exec, graph2, 0));
}

CUgraph graph1;
CUgraphExec graph1Exec;
void ConstructGraph1(void)
{
#define SIZE    (0x2UL << 20)
    CUdeviceptr d_a, d_b;
    int *h_a, *h_b;
    CUgraphNode mNode1, kNode1, mNode2;
    CUDA_KERNEL_NODE_PARAMS kNodeParams = {0};
    CUDA_MEMCPY3D mParams1 = {0};
    CUDA_MEMCPY3D mParams2 = {0};

    CHECK(cuMemAlloc(&d_a, SIZE));
    CHECK(cuMemAlloc(&d_b, SIZE));
    CHECK(cuMemAllocHost((void **)&h_a, SIZE));
    CHECK(cuMemAllocHost((void **)&h_b, SIZE));

    mParams1.Depth = 1;
    mParams1.Height = 1;
    mParams1.WidthInBytes = SIZE;
    mParams1.dstDevice = d_a;
    mParams1.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    mParams1.srcHost = h_a;
    mParams1.srcMemoryType = CU_MEMORYTYPE_HOST;

    mParams2.Depth = 1;
    mParams2.Height = 1;
    mParams2.WidthInBytes = SIZE;
    mParams2.dstHost = h_b;
    mParams2.dstMemoryType = CU_MEMORYTYPE_HOST;
    mParams2.srcDevice = d_b;
    mParams2.srcMemoryType = CU_MEMORYTYPE_DEVICE;

    void *params[] = {&d_a, &d_b};
    kNodeParams.blockDimX = 1;
    kNodeParams.blockDimY = 1;
    kNodeParams.blockDimZ = 1;
    kNodeParams.func = calcu;
    kNodeParams.gridDimX = 1;
    kNodeParams.gridDimY = 1;
    kNodeParams.gridDimZ = 1;
    kNodeParams.kernelParams = params;

    CHECK(cuGraphCreate(&graph1, 0));
    CHECK(cuGraphAddMemcpyNode(&mNode1, graph1,
                NULL, 0, &mParams1, context));
    CHECK(cuGraphAddKernelNode(&kNode1, graph1,
                &mNode1, 1, &kNodeParams));
    CHECK(cuGraphAddMemcpyNode(&mNode2, graph1,
                &kNode1, 1, &mParams2, context));

    CHECK(cuGraphInstantiate(&graph1Exec, graph1, 0));
}

void Graph1CompareTest(int cycle_num, CUstream stream)
{
    struct timeval start, end;
    CUdeviceptr d_a, d_b;
    int *h_a, *h_b;
    float sw_time = 0;
    void *params[] = {&d_a, &d_b};

    CHECK(cuMemAlloc(&d_a, SIZE));
    CHECK(cuMemAlloc(&d_b, SIZE));
    CHECK(cuMemAllocHost((void **)&h_a, SIZE));
    CHECK(cuMemAllocHost((void **)&h_b, SIZE));

    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&start, NULL);
        CHECK(cuMemcpyHtoDAsync(d_a, h_a, SIZE, stream));
        CHECK(cuLaunchKernel(calcu,
            1, 1, 1,
            1, 1, 1,
            0, stream, params, NULL));
        CHECK(cuMemcpyDtoHAsync(h_b, d_b, SIZE, stream));
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&end, NULL);

        sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
            (start.tv_sec * 1000000 + start.tv_usec);
        ResArray[i] = sw_time;
    }
    printf("stream exec avg time = %f us\n", Avg(ResArray, cycle_num));
    printf("max time = %f us\n", Max(ResArray, cycle_num));
    printf("min time = %f us\n", Min(ResArray, cycle_num));
    printf("std time = %f us\n", Std(ResArray, cycle_num));
}

void GraphPerfTest4(int cycle_num)
{
    CUstream stream;
    CUdeviceptr d_a, d_b;
    void *params[] = {&d_a, &d_b};
    struct timeval time[4];
    float st_host, st_dev, gr_host, gr_dev;

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuMemAlloc(&d_a, SIZE));
    CHECK(cuMemAlloc(&d_b, SIZE));

    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    /* stream test */
    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&time[0], NULL);
        for (int j = 0; j < 32; j++) {
            CHECK(cuLaunchKernel(fast,
                1, 1, 1,
                1, 1, 1,
                0, stream, params, NULL));
        }
        gettimeofday(&time[1], NULL);
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&time[2], NULL);

        HostTime[i] = __cal_time(&time[0], &time[1]);
        DevTime[i] = __cal_time(&time[0], &time[2]);
    }
    st_host = Avg(HostTime, cycle_num);
    st_dev = Avg(DevTime, cycle_num);
    printf("stream striaght line host = %f us\n", st_host);
    printf("stream striaght line dev = %f us\n", st_dev);

    ConstructStriaghtLine();
    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&time[0], NULL);
        CHECK(cuGraphLaunch(graphSLExec, stream));
        gettimeofday(&time[1], NULL);
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&time[2], NULL);

        HostTime[i] = __cal_time(&time[0], &time[1]);
        DevTime[i] = __cal_time(&time[0], &time[2]);
    }
    gr_host = Avg(HostTime, cycle_num);
    gr_dev = Avg(DevTime, cycle_num);
    printf("graph striaght line host = %f us\n", gr_host);
    printf("graph striaght line dev = %f us\n", gr_dev);

    printf("striaght line host acc %f\n", st_host / gr_host);
    printf("striaght line dev acc %f\n", st_dev / gr_dev);
}

void GraphPerfTest3(int cycle_num)
{
    CUstream stream;
    CUdeviceptr d_a, d_b;
    void *params[] = {&d_a, &d_b};
    struct timeval time[4];
    float st_host, st_dev, gr_host, gr_dev;

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuMemAlloc(&d_a, SIZE));
    CHECK(cuMemAlloc(&d_b, SIZE));

    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    /* stream test */
    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&time[0], NULL);
        for (int j = 0; j < 32; j++) {
            CHECK(cuLaunchKernel(fast,
                1, 1, 1,
                1, 1, 1,
                0, stream, params, NULL));
        }
        gettimeofday(&time[1], NULL);
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&time[2], NULL);

        HostTime[i] = __cal_time(&time[0], &time[1]);
        DevTime[i] = __cal_time(&time[0], &time[2]);
    }
    st_host = Avg(HostTime, cycle_num);
    st_dev = Avg(DevTime, cycle_num);
    printf("stream striaght line host = %f us\n", st_host);
    printf("stream striaght line dev = %f us\n", st_dev);

    ConstructStriaghtLine();
    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&time[0], NULL);
        CHECK(cuGraphLaunch(graphSLExec, stream));
        gettimeofday(&time[1], NULL);
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&time[2], NULL);

        HostTime[i] = __cal_time(&time[0], &time[1]);
        DevTime[i] = __cal_time(&time[0], &time[2]);
    }
    gr_host = Avg(HostTime, cycle_num);
    gr_dev = Avg(DevTime, cycle_num);
    printf("graph striaght line host = %f us\n", gr_host);
    printf("graph striaght line dev = %f us\n", gr_dev);

    printf("striaght line host acc %f\n", st_host / gr_host);
    printf("striaght line dev acc %f\n", st_dev / gr_dev);
}

void GraphPerfTest2(int cycle_num)
{
    CUstream stream;
    struct timeval start, end;
    float sw_time = 0;

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    /* stream test */
    Graph1CompareTest(cycle_num, stream);

    ConstructGraph2();
    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&start, NULL);
        CHECK(cuGraphLaunch(graph2Exec, stream));
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&end, NULL);

        sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
            (start.tv_sec * 1000000 + start.tv_usec);
        ResArray[i] = sw_time;
    }
    printf("graph2 exec avg time = %f us\n", Avg(ResArray, cycle_num));
    printf("max time = %f us\n", Max(ResArray, cycle_num));
    printf("min time = %f us\n", Min(ResArray, cycle_num));
    printf("std time = %f us\n", Std(ResArray, cycle_num));
}

void GraphPerfTest1(int cycle_num)
{
    CUstream stream;
    struct timeval start, end;
    float sw_time = 0;

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    /* stream test */
    Graph1CompareTest(cycle_num, stream);

    ConstructGraph1();

    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&start, NULL);
        CHECK(cuGraphLaunch(graph1Exec, stream));
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&end, NULL);

        sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
            (start.tv_sec * 1000000 + start.tv_usec);
        ResArray[i] = sw_time;
    }

    printf("graph1 exec avg time = %f us\n", Avg(ResArray, cycle_num));
    printf("max time = %f us\n", Max(ResArray, cycle_num));
    printf("min time = %f us\n", Min(ResArray, cycle_num));
    printf("std time = %f us\n", Std(ResArray, cycle_num));
}


volatile int flag = 0;
void StartFunc(void *data)
{
    while (1) {
        if (__sync_fetch_and_add(&flag, 0)) {
//            printf("host function return\n");
            return;
        }
        usleep(2);
    }
}

void TaskSwitchCost(int cycle_num)
{
#define DMA_SIZE    (0x2UL << 20)
    CUstream stream;
    CUdeviceptr d_a;
    int *h_a;
    struct timeval start, end;
    float sw_time1 = 0, sw_time2 = 0;
    void *params[] = {&d_a};

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CHECK(cuMemAlloc(&d_a, DMA_SIZE));
    CHECK(cuMemAllocHost((void **)&h_a, DMA_SIZE));

    flag = 0;
    CHECK(cuLaunchHostFunc(stream, StartFunc, NULL));
    for (int i = 0; i < cycle_num; i++) {
        CHECK(cuMemcpyHtoDAsync(d_a, h_a, DMA_SIZE, stream));
    }
    for (int i = 0; i < cycle_num; i++) {
        CHECK(cuLaunchKernel(slow,
            1, 1, 1,
            1, 1, 1,
            0, stream, params, NULL));
    }
    gettimeofday(&start, NULL);
    __sync_fetch_and_add(&flag, 1);
    CHECK(cuStreamSynchronize(stream));
    gettimeofday(&end, NULL);

    sw_time1 = (end.tv_sec * 1000000 + end.tv_usec) -
        (start.tv_sec * 1000000 + start.tv_usec);
    printf("KKKK+DDDD total time = %f us\n", sw_time1);

    flag = 0;
    CHECK(cuLaunchHostFunc(stream, StartFunc, NULL));
    for (int i = 0; i < cycle_num; i++) {
        CHECK(cuMemcpyHtoDAsync(d_a, h_a, DMA_SIZE, stream));
        CHECK(cuLaunchKernel(slow,
            1, 1, 1,
            1, 1, 1,
            0, stream, params, NULL));
    }
    gettimeofday(&start, NULL);
    __sync_fetch_and_add(&flag, 1);
    CHECK(cuStreamSynchronize(stream));
    gettimeofday(&end, NULL);

    sw_time2 = (end.tv_sec * 1000000 + end.tv_usec) -
        (start.tv_sec * 1000000 + start.tv_usec);
    printf("KDKDKDKD total time = %f us\n", sw_time2);
    printf("substruction = %f, swtich time = %f\n",
            sw_time2 - sw_time1, (sw_time2 - sw_time1)/(cycle_num*2-2));
}

void LaunchDelay(int type, int cycle_num)
{
    CUstream stream;
    CUdeviceptr d_a;
    int *h_a;
    struct timeval start, end;
    float sw_time = 0;
    void *params[] = {&d_a};

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CHECK(cuMemAlloc(&d_a, 1024));
    CHECK(cuMemAllocHost((void **)&h_a, 1024));

    for (int i = 0; i < cycle_num; i++) {
        gettimeofday(&start, NULL);
    	switch (type) {
    	case 0: {
            CHECK(cuLaunchKernel(fast,
                1, 1, 1,
                1, 1, 1,
                0, stream, params, NULL));
    		break;
    	}

    	case 1: {
            CHECK(cuMemcpyDtoHAsync(h_a, d_a, 1024, stream));
    		break;
    	}

    	case 2: {
            CHECK(cuLaunchHostFunc(stream, __HostFunc, NULL));
    		break;
    	}
    	default: {
    		printf("Unknown Type\n");
    		break;
        }
    	}
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&end, NULL);
    	sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
    	    (start.tv_sec * 1000000 + start.tv_usec);
        ResArray[i] = sw_time;
    }

    printf("average launch sync time = %f us\n", Avg(ResArray, cycle_num));
    printf("max time = %f us\n", Max(ResArray, cycle_num));
    printf("min time = %f us\n", Min(ResArray, cycle_num));
    printf("std time = %f us\n", Std(ResArray, cycle_num));
}

void PipeLineTime(int type, int cycle_num)
{
    CUstream stream;
    CUdeviceptr d_a;
    int *h_a;
    struct timeval start, end;
    float sw_time = 0;
    void *params[] = {&d_a};

    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    CHECK(cuMemAlloc(&d_a, 1024));
    CHECK(cuMemAllocHost((void **)&h_a, 1024));

    gettimeofday(&start, NULL);
    for (int i = 0; i < cycle_num; i++) {
        switch (type) {
        case 0: {
            CHECK(cuLaunchKernel(fast,
                1, 1, 1,
                1, 1, 1,
                0, stream, params, NULL));
            break;
        }
        case 1: {
            CHECK(cuMemcpyDtoHAsync(h_a, d_a, 1024, stream));
            break;
        }
        case 2: {
            CHECK(cuLaunchHostFunc(stream, __HostFunc, NULL));
            break;
        }
        default:
            printf("Unknown Task Type!\n");
            break;
        }
    }

    CHECK(cuStreamSynchronize(stream));
    gettimeofday(&end, NULL);

    sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
        (start.tv_sec * 1000000 + start.tv_usec);

    printf("total time = %f us, single time = %f us\n", sw_time, sw_time/cycle_num);
}

int main(int argc, char **argv)
{
    int mode = 0, type = 0, cycle = 0;

    if (argc != 4) {
        printf("Args1 Test Mode: \n");
        printf("0 PipeLineTest\n");
        printf("1 LaunchDelayTest\n");
        printf("2 TaskSwitchCostTest\n");
        printf("3 Graph1Test\n");
        printf("4 Graph2Test\n");
        printf("5 Graph3Test\n");

        printf("Args2 Chose Task Type\n");
        printf("0 Kernel\n");
        printf("1 AsyncDMA\n");
        printf("2 HostFunc\n");

        printf("Args3 Please Input Cycle Number\n");
        return 1;
    }

    mode = atoi(argv[1]);
    type = atoi(argv[2]);
    cycle = atoi(argv[3]);
    printf("mode[%d]type[%d]cycle[%d]\n", mode, type, cycle);

    ResArray = (float *)calloc(cycle, sizeof(float));
    HostTime = (float *)calloc(cycle, sizeof(float));
    DevTime = (float *)calloc(cycle, sizeof(float));

    printf("- Init...\n");
    initCUDA();

    switch (mode) {
        case 0: {
            PipeLineTime(type, cycle);
            break;
        }
        case 1: {
            LaunchDelay(type, cycle);
            break;
        }
        case 2: {
            TaskSwitchCost(cycle);
            break;
        }
        case 3: {
            GraphPerfTest1(cycle);
            break;
        }
        case 4: {
            GraphPerfTest2(cycle);
            break;
        }
        case 5: {
            GraphPerfTest3(cycle);
            break;
        }
        default:
            printf("Error Mode!\n");
            break;
    }
    exitCUDA();
    return 0;
}
