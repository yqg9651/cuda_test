#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/time.h>

#define CHECK(func) \
{   \
    if (func != CUDA_SUCCESS) { \
        printf("Function Call %s Failed!\n", #func);    \
        exit(1);    \
    }   \
}

#define SIZE		(20UL << 20)
CUdeviceptr data1, data2;
CUdevice device;
CUcontext context1, context2;
CUmodule module1, module2;
CUfunction function1, function2;

volatile int reset_flag = 0;
volatile int done_flag = 0;
volatile int seq_flag = 0;

#define LAUNCH(func)	\
{	\
	gettimeofday(&start, NULL);	\
        CHECK(cuLaunchKernel(func,	\
            1, 1, 1,	\
            1, 1, 1,	\
            0, stream, params, NULL));	\
   	CHECK(cuStreamSynchronize(stream));	\
	gettimeofday(&end, NULL);	\
	\
	sw_time = (end.tv_sec * 1000000 + end.tv_usec) -	\
		(start.tv_sec * 1000000 + start.tv_usec);	\
	printf("%d l2 cache sw_time = %f\n", tid, sw_time);	\
}

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
    CHECK(cuCtxCreate(&context1, 0, device));
    CHECK(cuModuleLoad(&module1, "kernel.ptx"));
    CHECK(cuModuleGetFunction(&function1, module1, "TestKernel"));

    CHECK(cuCtxCreate(&context2, 0, device));
    CHECK(cuModuleLoad(&module2, "kernel.ptx"));
    CHECK(cuModuleGetFunction(&function2, module2, "TestKernel"));
}

void *ThreadFunc1(void *data)
{
	cudaStream_t stream;
	struct timeval start, end;
	float sw_time = 0;
	int size = SIZE;
	void *params[] = {&size, &data1};
	int tid = 1;

    	CHECK(cuCtxSetCurrent(context1));

    	CHECK(cuMemAlloc(&data1, SIZE));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

	/* set L2 persisting cache */
	CUstreamAttrValue attr;
	attr.accessPolicyWindow.base_ptr = (void *)data1;
	attr.accessPolicyWindow.num_bytes = SIZE;
	attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
	attr.accessPolicyWindow.hitRatio = 1;
	attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;

//	CHECK(cuStreamSetAttribute(stream,
//		CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));

        CHECK(cuLaunchKernel(function1,
            1, 1, 1,
            1, 1, 1,
            0, stream, params, NULL));
   	CHECK(cuStreamSynchronize(stream));

	LAUNCH(function1);
	LAUNCH(function1);
	__sync_fetch_and_add(&seq_flag, 1);

	while (!__sync_fetch_and_add(&done_flag, 0))
		sleep(1);

	//cuCtxResetPersistingL2Cache();
   	CHECK(cuCtxSynchronize());
	printf("after reset\n");
	__sync_fetch_and_add(&reset_flag, 1);

	return NULL;
}

void *ThreadFunc2(void *data)
{
	cudaStream_t stream;
	struct timeval start, end;
	float sw_time = 0;
	int size = SIZE;
	void *params[] = {&size, &data2};
	int tid = 2;

    	CHECK(cuCtxSetCurrent(context2));

    	CHECK(cuMemAlloc(&data2, SIZE));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

	/* set L2 persisting cache */
	CUstreamAttrValue attr;
	attr.accessPolicyWindow.base_ptr = (void *)data2;
	attr.accessPolicyWindow.num_bytes = SIZE;
	attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
	attr.accessPolicyWindow.hitRatio = 1;
	attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;

	CHECK(cuStreamSetAttribute(stream,
		CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));

	while (!__sync_fetch_and_add(&seq_flag, 0))
		sleep(1);

        CHECK(cuLaunchKernel(function2,
            1, 1, 1,
            1, 1, 1,
            0, stream, params, NULL));
   	CHECK(cuStreamSynchronize(stream));

	LAUNCH(function2);
	LAUNCH(function2);

	__sync_fetch_and_add(&done_flag, 1);

	while (!__sync_fetch_and_add(&reset_flag, 0))
		sleep(1);

	/* after reset test */
	LAUNCH(function2);

	return NULL;
}

int main(int argc, char **argv)
{
	pthread_t id1, id2;

	printf("- Init...\n");
	initCUDA();

	CHECK(pthread_create(&id2, NULL, ThreadFunc2, NULL));
	CHECK(pthread_create(&id1, NULL, ThreadFunc1, NULL));

	CHECK(pthread_join(id1, NULL));
	CHECK(pthread_join(id2, NULL));

    return 0;
}
