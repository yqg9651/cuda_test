#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#define CHECK(func) \
{   \
    if (func != CUDA_SUCCESS) { \
        printf("Function Call %s Failed!\n", #func);    \
        exit(1);    \
    }   \
}

#define SIZE		(20UL << 20)
CUdeviceptr data;
CUdevice device;
CUcontext context;
CUmodule module;
CUfunction function;

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

#define GET_VALUE(val)	\
{	\
	int tmp = 0;	\
	do {	\
		tmp = semctl(sem_id, 0, GETVAL);	\
	} while (tmp != val);	\
}

#define SET_VALUE(val)	\
{	\
	semctl(sem_id, 0, SETVAL, val);	\
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
    CHECK(cuCtxCreate(&context, 0, device));
    CHECK(cuModuleLoad(&module, "kernel.ptx"));
    CHECK(cuModuleGetFunction(&function, module, "TestKernel"));
}

void ProcessFunc1(void)
{
	CUstream stream;
	struct timeval start, end;
	float sw_time = 0;
	int size = SIZE;
	void *params[] = {&size, &data};
	int tid = 1;
	key_t key;
	int sem_id;

	key = ftok("abc", 0xFF);
	sem_id = semget(key, 1, IPC_CREAT|0644);

	SET_VALUE(321);

	printf("- Process1 Init...\n");
	initCUDA();

    	CHECK(cuCtxSetCurrent(context));

    	CHECK(cuMemAlloc(&data, SIZE));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

	/* set L2 persisting cache */
	CUstreamAttrValue attr;
	attr.accessPolicyWindow.base_ptr = (void *)data;
	attr.accessPolicyWindow.num_bytes = SIZE;
	attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
	attr.accessPolicyWindow.hitRatio = 1;
	attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;

	CHECK(cuStreamSetAttribute(stream,
		CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));

//        CHECK(cuLaunchKernel(function,
//            1, 1, 1,
//            1, 1, 1,
//            0, stream, params, NULL));
//   	CHECK(cuStreamSynchronize(stream));

	LAUNCH(function);
	LAUNCH(function);
	LAUNCH(function);
	LAUNCH(function);
	
	SET_VALUE(1);

	GET_VALUE(2);

	//cuCtxResetPersistingL2Cache();
   	CHECK(cuCtxSynchronize());
	printf("after reset\n");

	SET_VALUE(3);

	return;
}

void ProcessFunc2(void)
{
	CUstream stream;
	struct timeval start, end;
	float sw_time = 0;
	int size = SIZE;
	void *params[] = {&size, &data};
	int tid = 2;
	key_t key;
	int sem_id;

	key = ftok("abc", 0xFF);
	sem_id = semget(key, 1, IPC_CREAT|0644);

	printf("- Process2 Init...\n");
	initCUDA();

    	CHECK(cuCtxSetCurrent(context));

    	CHECK(cuMemAlloc(&data, SIZE));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

	/* set L2 persisting cache */
	CUstreamAttrValue attr;
	attr.accessPolicyWindow.base_ptr = (void *)data;
	attr.accessPolicyWindow.num_bytes = SIZE;
	attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
	attr.accessPolicyWindow.hitRatio = 1;
	attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;

	CHECK(cuStreamSetAttribute(stream,
		CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));

	GET_VALUE(1);

//        CHECK(cuLaunchKernel(function,
//            1, 1, 1,
//            1, 1, 1,
//            0, stream, params, NULL));
//   	CHECK(cuStreamSynchronize(stream));

	LAUNCH(function);
	LAUNCH(function);
	LAUNCH(function);
	LAUNCH(function);

	SET_VALUE(2);

	GET_VALUE(3);

//	attr.accessPolicyWindow.base_ptr = (void *)data;
//	attr.accessPolicyWindow.num_bytes = SIZE;
//	attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_NORMAL;
//	attr.accessPolicyWindow.hitRatio = 1;
//	attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_NORMAL;
//
//	CHECK(cuStreamSetAttribute(stream,
//		CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
	/* after reset test */
//	LAUNCH(function);
	LAUNCH(function);

	return;
}

int main(int argc, char **argv)
{
	pid_t child;
	int status;
	
	child = fork();

	if (child) {
		/* parent */
		ProcessFunc1();
		wait(&status);
	} else {
		/* child */
		sleep(5);
		ProcessFunc2();
	}


    return 0;
}
