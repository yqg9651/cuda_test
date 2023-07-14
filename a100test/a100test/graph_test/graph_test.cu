#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include "math.h"
#include "cuda.h"

#define STREAM_LESS     1

#define CHECK(func) \
do {    \
    int ret = 0;	\
    if ((ret = func) != CUDA_SUCCESS) {   \
        printf("function %s = %d Failed\n", #func, ret);   \
    }   \
} while (0)

double Avg(double *array, int len)
{
    double sum = 0;
    for (int i = 0; i < len; i++)
        sum += array[i];

    return sum / len;
}

CUdevice device;
CUcontext context;
CUmodule mod;
CUfunction fast;
double *HostTime;
double *DevTime;

void initCUDA(void)
{
    int devCount = 0;

    CHECK(cuInit(0));
    CHECK(cuDeviceGetCount(&devCount));
    CHECK(devCount == 0);
    CHECK(cuCtxCreate(&context, 0, device));
    CHECK(cuModuleLoad(&mod, "test.ptx"));
    CHECK(cuModuleGetFunction(&fast, mod, "FastKernel"));
}

void exitCUDA(void)
{
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(context));
}

double __cal_time(struct timeval *start, struct timeval *end)
{
    return (end->tv_sec * 1000000 + end->tv_usec) -
            (start->tv_sec * 1000000 + start->tv_usec);
}

/***************************************************/
struct timeval _time[4];
double stream_host, stream_dev, graph_host, graph_dev;
#define NODE_NUM    (32U)

CUgraph g_fork_and_join;
CUgraphExec g_FJ_exec;
void ConstructForkJoin(void)
{
    CUgraphNode kNode[NODE_NUM];
    CUDA_KERNEL_NODE_PARAMS kNodeParams = {0};
    int i = 0;

    kNodeParams.blockDimX = 1;
    kNodeParams.blockDimY = 1;
    kNodeParams.blockDimZ = 1;
    kNodeParams.gridDimX = 1;
    kNodeParams.gridDimY = 1;
    kNodeParams.gridDimZ = 1;
    kNodeParams.func = fast;
    kNodeParams.kernelParams = NULL;

    CHECK(cuGraphCreate(&g_fork_and_join, 0));

    CUgraphNode *u_depend = NULL;
    CUgraphNode depend[2] = {NULL, NULL};
    int u_num = 0;

    for (i = 0; i < 30; i++) {
        if (i == 0) {
            u_depend = NULL;
            u_num = 0;
        } else {
            depend[0] = kNode[i * 3 - 1];
            depend[1] = kNode[i * 3 - 2];
            u_depend = depend;
            u_num = 2;
        }
        CHECK(cuGraphAddKernelNode(&kNode[i * 3], g_fork_and_join, u_depend, u_num, &kNodeParams));

        if (i == 10)
            break;

        CHECK(cuGraphAddKernelNode(&kNode[i * 3 + 1], g_fork_and_join, &kNode[i * 3], 1, &kNodeParams));
        CHECK(cuGraphAddKernelNode(&kNode[i * 3 + 2], g_fork_and_join, &kNode[i * 3], 1, &kNodeParams));
    }

    CHECK(cuGraphInstantiate(&g_FJ_exec, g_fork_and_join, NULL, NULL, 0));
}

void GraphForkAndJoin(int cycle)
{
    CUstream s1, s2, s3;
    CUevent ev1, ev2, ev3;
    int i = 0;
    
    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
    CHECK(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));
    CHECK(cuStreamCreate(&s3, CU_STREAM_NON_BLOCKING));
    CHECK(cuEventCreate(&ev1, 0));
    CHECK(cuEventCreate(&ev2, 0));
    CHECK(cuEventCreate(&ev3, 0));

    /* stream test */
#ifndef STREAM_LESS
    for (int j = 0; j < cycle; j++) {
        gettimeofday(&_time[0], NULL);
        for (i = 0; i < 30; i++) {
            if (i != 0) {
                CHECK(cuStreamWaitEvent(s1, ev2, 0));
                CHECK(cuStreamWaitEvent(s1, ev3, 0));
            }

            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));

            if (i == 10)
                break;

            CHECK(cuEventRecord(ev1, s1));

            CHECK(cuStreamWaitEvent(s2, ev1, 0));
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s2, NULL, NULL));
            CHECK(cuEventRecord(ev2, s2));

            CHECK(cuStreamWaitEvent(s3, ev1, 0));
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s3, NULL, NULL));
            CHECK(cuEventRecord(ev3, s3));
        }
        gettimeofday(&_time[1], NULL);

        CHECK(cuStreamSynchronize(s1));
        CHECK(cuStreamSynchronize(s2));
        CHECK(cuStreamSynchronize(s3));
        gettimeofday(&_time[2], NULL);

        HostTime[j] = __cal_time(&_time[0], &_time[1]);
        DevTime[j] = __cal_time(&_time[0], &_time[2]);
    }
#else
    for (int j = 0; j < cycle; j++) {
        gettimeofday(&_time[0], NULL);
        for (i = 0; i < 30; i++) {
            if (i != 0) {
                CHECK(cuStreamWaitEvent(s1, ev2, 0));
            }
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));

            if (i == 10)
                break;

            CHECK(cuEventRecord(ev1, s1));
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));

            CHECK(cuStreamWaitEvent(s2, ev1, 0));
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s2, NULL, NULL));
            CHECK(cuEventRecord(ev2, s2));
        }
        gettimeofday(&_time[1], NULL);

        CHECK(cuStreamSynchronize(s1));
        CHECK(cuStreamSynchronize(s2));
        gettimeofday(&_time[2], NULL);

        HostTime[j] = __cal_time(&_time[0], &_time[1]);
        DevTime[j] = __cal_time(&_time[0], &_time[2]);
    }
#endif

    stream_host = Avg(HostTime, cycle);
    stream_dev = Avg(DevTime, cycle);

    printf("cycle[%d] stream fork and join host time %f us\n", cycle, stream_host);
    printf("cycle[%d] stream fork and join dev time %f us\n", cycle, stream_dev);

    /*  graph test*/
    ConstructForkJoin();

    for (int i = 0; i < cycle; i++) {
        gettimeofday(&_time[0], NULL);
        CHECK(cuGraphLaunch(g_FJ_exec, s1));
        gettimeofday(&_time[1], NULL);
        CHECK(cuStreamSynchronize(s1));
        gettimeofday(&_time[2], NULL);

        HostTime[i] = __cal_time(&_time[0], &_time[1]);
        DevTime[i] = __cal_time(&_time[0], &_time[2]);
    }

    graph_host = Avg(HostTime, cycle);
    graph_dev = Avg(DevTime, cycle);

    printf("cycle[%d] graph fork and join host time %f us\n", cycle, graph_host);
    printf("cycle[%d] graph fork and join dev time %f us\n", cycle, graph_dev);

    printf("host ratio %f\n", stream_host / graph_host);
    printf("dev ratio %f\n", stream_dev / graph_dev);

    CHECK(cuEventDestroy(ev1));
    CHECK(cuEventDestroy(ev2));
    CHECK(cuEventDestroy(ev3));
    CHECK(cuStreamDestroy(s1));
    CHECK(cuStreamDestroy(s2));
    CHECK(cuStreamDestroy(s3));
}

CUgraph g_two_branches;
CUgraphExec g_TB_exec;
void ConstructTwoBranches(void)
{
    CUgraphNode kNode[NODE_NUM];
    CUDA_KERNEL_NODE_PARAMS kNodeParams = {0};

    kNodeParams.blockDimX = 1;
    kNodeParams.blockDimY = 1;
    kNodeParams.blockDimZ = 1;
    kNodeParams.gridDimX = 1;
    kNodeParams.gridDimY = 1;
    kNodeParams.gridDimZ = 1;
    kNodeParams.func = fast;
    kNodeParams.kernelParams = NULL;

    CHECK(cuGraphCreate(&g_two_branches, 0));

    CHECK(cuGraphAddKernelNode(&kNode[0], g_two_branches, NULL, 0, &kNodeParams));
    CHECK(cuGraphAddKernelNode(&kNode[1], g_two_branches, &kNode[0], 1, &kNodeParams));
    CHECK(cuGraphAddKernelNode(&kNode[16], g_two_branches, &kNode[0], 1, &kNodeParams));

    for (int i = 2; i < 2 + (NODE_NUM - 4) / 2; i++) {
        CHECK(cuGraphAddKernelNode(&kNode[i], g_two_branches, &kNode[i - 1], 1, &kNodeParams));
    }

    for (int i = 17; i < 17 + (NODE_NUM - 4) / 2; i++) {
        CHECK(cuGraphAddKernelNode(&kNode[i], g_two_branches, &kNode[i - 1], 1, &kNodeParams));
    }

    CUgraphNode depend[2] = {kNode[15], kNode[30]};
    CHECK(cuGraphAddKernelNode(&kNode[31], g_two_branches, depend, 2, &kNodeParams));

    CHECK(cuGraphInstantiate(&g_TB_exec, g_two_branches, NULL, NULL, 0));
}

void GraphTwoBranches(int cycle)
{
    CUstream s1, s2, s3;
    CUevent ev1, ev2, ev3;
    
    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
    CHECK(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));
    CHECK(cuStreamCreate(&s3, CU_STREAM_NON_BLOCKING));
    CHECK(cuEventCreate(&ev1, 0));
    CHECK(cuEventCreate(&ev2, 0));
    CHECK(cuEventCreate(&ev3, 0));

    /* stream test */
#ifndef STREAM_LESS
    for (int j = 0; j < cycle; j++) {
        gettimeofday(&_time[0], NULL);
        CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));
        CHECK(cuEventRecord(ev1, s1));

        CHECK(cuStreamWaitEvent(s2, ev1, 0));
        for (int i = 0; i < 15; i++) {
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s2, NULL, NULL));
        }
        CHECK(cuEventRecord(ev2, s2));

        CHECK(cuStreamWaitEvent(s3, ev1, 0));
        for (int i = 0; i < 15; i++) {
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s3, NULL, NULL));
        }
        CHECK(cuEventRecord(ev3, s3));
   
        CHECK(cuStreamWaitEvent(s1, ev2, 0));
        CHECK(cuStreamWaitEvent(s1, ev3, 0));
        CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));
        gettimeofday(&_time[1], NULL);

        CHECK(cuStreamSynchronize(s1));
        CHECK(cuStreamSynchronize(s2));
        CHECK(cuStreamSynchronize(s3));
        gettimeofday(&_time[2], NULL);

        HostTime[j] = __cal_time(&_time[0], &_time[1]);
        DevTime[j] = __cal_time(&_time[0], &_time[2]);
    }
#else
    for (int j = 0; j < cycle; j++) {
        gettimeofday(&_time[0], NULL);

        CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));
        CHECK(cuEventRecord(ev1, s1));
        for (int i = 0; i < 15; i++) {
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));
        }

        CHECK(cuStreamWaitEvent(s2, ev1, 0));
        for (int i = 0; i < 15; i++) {
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s2, NULL, NULL));
        }
        CHECK(cuEventRecord(ev2, s2));

        CHECK(cuStreamWaitEvent(s1, ev2, 0));
        CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, s1, NULL, NULL));
        gettimeofday(&_time[1], NULL);

        CHECK(cuStreamSynchronize(s1));
        CHECK(cuStreamSynchronize(s2));
        gettimeofday(&_time[2], NULL);

        HostTime[j] = __cal_time(&_time[0], &_time[1]);
        DevTime[j] = __cal_time(&_time[0], &_time[2]);
    }
#endif

    stream_host = Avg(HostTime, cycle);
    stream_dev = Avg(DevTime, cycle);

    printf("cycle[%d] stream two branches host time %f us\n", cycle, stream_host);
    printf("cycle[%d] stream two branches dev time %f us\n", cycle, stream_dev);

    /*  graph test*/
    ConstructTwoBranches();

    for (int i = 0; i < cycle; i++) {
        gettimeofday(&_time[0], NULL);
        CHECK(cuGraphLaunch(g_TB_exec, s1));
        gettimeofday(&_time[1], NULL);
        CHECK(cuStreamSynchronize(s1));
        gettimeofday(&_time[2], NULL);

        HostTime[i] = __cal_time(&_time[0], &_time[1]);
        DevTime[i] = __cal_time(&_time[0], &_time[2]);
    }

    graph_host = Avg(HostTime, cycle);
    graph_dev = Avg(DevTime, cycle);

    printf("cycle[%d] graph two branches host time %f us\n", cycle, graph_host);
    printf("cycle[%d] graph two branches dev time %f us\n", cycle, graph_dev);

    printf("host ratio %f\n", stream_host / graph_host);
    printf("dev ratio %f\n", stream_dev / graph_dev);

    CHECK(cuEventDestroy(ev1));
    CHECK(cuEventDestroy(ev2));
    CHECK(cuEventDestroy(ev3));
    CHECK(cuStreamDestroy(s1));
    CHECK(cuStreamDestroy(s2));
    CHECK(cuStreamDestroy(s3));
}

CUgraph g_striaght_line;
CUgraphExec g_SL_exec;
void ConstructStriaghtLine(void)
{
    CUgraphNode kNode[NODE_NUM];
    CUDA_KERNEL_NODE_PARAMS kNodeParams = {0};

    kNodeParams.blockDimX = 1;
    kNodeParams.blockDimY = 1;
    kNodeParams.blockDimZ = 1;
    kNodeParams.gridDimX = 1;
    kNodeParams.gridDimY = 1;
    kNodeParams.gridDimZ = 1;
    kNodeParams.func = fast;
    kNodeParams.kernelParams = NULL;

    CHECK(cuGraphCreate(&g_striaght_line, 0));

    CHECK(cuGraphAddKernelNode(&kNode[0], g_striaght_line, NULL, 0, &kNodeParams));
    for (int i = 1; i < NODE_NUM; i++) {
        CHECK(cuGraphAddKernelNode(&kNode[i], g_striaght_line, &kNode[i - 1], 1, &kNodeParams));
    }

    CHECK(cuGraphInstantiate(&g_SL_exec, g_striaght_line, NULL, NULL, 0));
}

void GraphStriaghtLine(int cycle)
{
    CUstream stream;
    
    CHECK(cuCtxSetCurrent(context));
    CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    /* stream test */
    for (int i = 0; i < cycle; i++) {
        gettimeofday(&_time[0], NULL);
        for (int j = 0; j < NODE_NUM; j++) {
            CHECK(cuLaunchKernel(fast, 1, 1, 1, 1, 1, 1, 0, stream, NULL, NULL));
        }
        gettimeofday(&_time[1], NULL);
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&_time[2], NULL);

        HostTime[i] = __cal_time(&_time[0], &_time[1]);
        DevTime[i] = __cal_time(&_time[0], &_time[2]);
    }

    stream_host = Avg(HostTime, cycle);
    stream_dev = Avg(DevTime, cycle);

    printf("cycle[%d] stream striaght line host time %f us\n", cycle, stream_host);
    printf("cycle[%d] stream striaght line dev time %f us\n", cycle, stream_dev);

    /*  graph test*/
    ConstructStriaghtLine();

    for (int i = 0; i < cycle; i++) {
        gettimeofday(&_time[0], NULL);
        CHECK(cuGraphLaunch(g_SL_exec, stream));
        gettimeofday(&_time[1], NULL);
        CHECK(cuStreamSynchronize(stream));
        gettimeofday(&_time[2], NULL);

        HostTime[i] = __cal_time(&_time[0], &_time[1]);
        DevTime[i] = __cal_time(&_time[0], &_time[2]);
    }

    graph_host = Avg(HostTime, cycle);
    graph_dev = Avg(DevTime, cycle);

    printf("cycle[%d] graph striaght line host time %f us\n", cycle, graph_host);
    printf("cycle[%d] graph striaght line dev time %f us\n", cycle, graph_dev);

    printf("host ratio %f\n", stream_host / graph_host);
    printf("dev ratio %f\n", stream_dev / graph_dev);

    CHECK(cuStreamDestroy(stream));
}

/***************************************************/
int main(int argc, char **argv)
{
    int cycle = 1000;

    HostTime = (double *)calloc(cycle, sizeof(double));
    DevTime = (double *)calloc(cycle, sizeof(double));

    printf("- Init...\n");
    initCUDA();

    printf("- Start Stright Line\n");
    GraphStriaghtLine(cycle);

    printf("- Start Two Branches\n");
    GraphTwoBranches(cycle);

    printf("- Start Fork and Join\n");
    GraphForkAndJoin(cycle);

    exitCUDA();
    return 0;
}
