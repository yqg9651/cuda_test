#include <stdio.h>
#include <pthread.h>
#include <cuda.h>
#include <sys/time.h>

volatile bool flag = 0;
int *data1, *data2;

#define SIZE		(20UL << 20)
#define LARGE_SIZE	(128UL << 20)
__global__ void cuda_kernelB(int size, int *data1)
{
	printf("size = %d\n", size);
	for (int i = 0; i < size/4 - 1; i++) {
		data1[i] = data1[i+1]+1;
	}
}

__global__ void loop(void)
{
	int a = 0;
	while (1) {
		a++;
	}
}

void *ThreadFunc1(void *data)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	struct timeval start, end;
	float sw_time = 0;

	cudaMalloc(&data1, SIZE);

	cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
	stream_attribute.accessPolicyWindow.base_ptr = (void *)data1; // Global Memory data pointer
	stream_attribute.accessPolicyWindow.num_bytes = SIZE; // Number of bytes for persistence access
	stream_attribute.accessPolicyWindow.hitRatio = 1; // Hint for cache hit ratio
	stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Persistence Property
	stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; // Type of access property on cache miss
	cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); // Overwrite the access policy attribute to a CUDA Stream

	/* warn up */
	//cuda_kernelB<<<1,1,0,stream>>>(SIZE, data1);
	loop<<<1,1,0>>>();
	//cudaStreamSynchronize(stream);
	cudaStreamSynchronize(NULL);
	printf("not sync!\n");

	gettimeofday(&start, NULL);
	cuda_kernelB<<<1,1,0,stream>>>(SIZE, data1);
	cudaStreamSynchronize(stream);
	gettimeofday(&end, NULL);

	sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
		(start.tv_sec * 1000000 + start.tv_usec);
	printf("l2 cache sw_time = %f\n", sw_time);

	loop<<<1,1,0,stream>>>();

	//cudaCtxResetPersistingL2Cache();
	printf("L2 reset done !\n");
	__sync_fetch_and_add(&flag, 1);

	cudaStreamSynchronize(stream);

	return NULL;
}

void *ThreadFunc2(void *data)
{
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	struct timeval start, end;
	float sw_time = 0;

	while (!__sync_fetch_and_add(&flag, 0))
		;

	printf("Start thread 2\n");
	cudaMalloc(&data2, LARGE_SIZE);

	cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
	stream_attribute.accessPolicyWindow.base_ptr = (void *)data1; // Global Memory data pointer
	stream_attribute.accessPolicyWindow.num_bytes = SIZE; // Number of bytes for persistence access
	stream_attribute.accessPolicyWindow.hitRatio = 1; // Hint for cache hit ratio
	stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Persistence Property
	stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; // Type of access property on cache miss
	cudaStreamSetAttribute(stream1, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); // Overwrite the access policy attribute to a CUDA Stream
	printf("set attribute success!\n");

	/* flush cache */
	cuda_kernelB<<<1,1,0,stream1>>>(LARGE_SIZE, data2);
	cudaStreamSynchronize(stream1);

	printf("sync success!\n");

	gettimeofday(&start, NULL);
	cuda_kernelB<<<1,1,0,stream1>>>(SIZE, data1);
	cudaStreamSynchronize(stream1);
	gettimeofday(&end, NULL);

	sw_time = (end.tv_sec * 1000000 + end.tv_usec) -
		(start.tv_sec * 1000000 + start.tv_usec);
	printf("after reset l2 cache sw_time = %f\n", sw_time);

	return NULL;
}

int main(int argc, char **argv)
{
	pthread_t id[2];

	pthread_create(&id[0], NULL, ThreadFunc1, NULL);
	pthread_create(&id[1], NULL, ThreadFunc2, NULL);

	pthread_join(id[0], NULL);
	pthread_join(id[1], NULL);

	return 0;
}
