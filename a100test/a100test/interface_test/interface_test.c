#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include "math.h"
#include "cuda.h"

#define CHECK(func)	\
do {	\
	int ret;	\
	if ((ret = (func)) != CUDA_SUCCESS) {	\
		printf("[%d] %s = %d Failed\n", __LINE__, #func, ret);	\
		exit(EXIT_FAILURE);	\
	}	\
} while (0)

struct interface_data {
	double avg;
	double max;
	double min;
	double std;
};

struct time_pack {
	struct timeval start;
	struct timeval end;
	double *result;
	int count;
	int index;
	struct interface_data data;
};

void cal_val(double *array, int num, struct interface_data *data)
{
	double sum = 0;
	double sqr = 0;
	int i;

	data->max = 0;
	data->min = 999999;
	for (i = 0; i < num; i++) {
		sum += array[i];

		if (data->max < array[i])
			data->max = array[i];

		if (data->min > array[i])
			data->min = array[i];
	}

	data->avg = sum / num;

	for (i = 0; i < num; i++) {
		sqr += pow((array[i] - data->avg), 2);
	}
	sqr /= num;
	data->std = pow(sqr, 0.5);
}

double cal_time(struct time_pack *pack)
{
	return ((pack->end.tv_sec * 1000000 + pack->end.tv_usec) - 
		(pack->start.tv_sec * 1000000 + pack->start.tv_usec)) / 1000.0;
}

void print_info(char *name, struct time_pack *pack)
{
	cal_val(pack->result, pack->count, &pack->data);
	printf("%s Call %d Times: \n", name, pack->count);
	printf("    Avg: %f ms\n", pack->data.avg);
	printf("    Max: %f ms\n", pack->data.max);
	printf("    Min: %f ms\n", pack->data.min);
	printf("    Std: %f ms\n", pack->data.std);

	free(pack->result);
	memset(pack, 0, sizeof(struct time_pack));
}

void pack_init(int count, struct time_pack *pack)
{
	memset(pack, 0, sizeof(struct time_pack));
	pack->count = count;
}

void time_start(struct time_pack *pack)
{
	if (!pack->result)
		pack->result = (double *)calloc(pack->count, sizeof(double));
	gettimeofday(&pack->start, NULL);
}

void time_end(struct time_pack *pack)
{
	gettimeofday(&pack->end, NULL);
	pack->result[pack->index++] = cal_time(pack);
}

CUdevice device;

void cal_time_execution(int count)
{
	int i;
	CUcontext ctx1, ctx2, str_ctx;
	CUstream stream;
	CUdeviceptr d_a;
	struct time_pack _create, _destroy, _current, _streamctx;

	pack_init(count, &_create);
	pack_init(count, &_destroy);
	pack_init(count, &_current);
	pack_init(count, &_streamctx);


	for (i = 0; i < count; i++) {
		time_start(&_create);
		CHECK(cuCtxCreate(&ctx1, 0, device));
		time_end(&_create);
//	CHECK(cuMemAlloc(&d_a, 1024));
		CHECK(cuCtxCreate(&ctx2, 0, device));

		time_start(&_current);
		CHECK(cuCtxSetCurrent(ctx1));
		time_end(&_current);

		CHECK(cuStreamCreate(&stream, 0));
		CHECK(cuCtxSetCurrent(ctx2));

		time_start(&_streamctx);
		CHECK(cuStreamGetCtx(stream, &str_ctx));
		time_end(&_streamctx);

	//	CHECK(cuStreamWaitValue32(stream, d_a, 0, CU_STREAM_WAIT_VALUE_EQ));

		CHECK(str_ctx != ctx1);
		CHECK(cuStreamDestroy(stream));

		time_start(&_destroy);
		CHECK(cuCtxDestroy(ctx1));
		time_end(&_destroy);
		CHECK(cuCtxDestroy(ctx2));
	}


	print_info("cuCtxCreate", &_create);
	print_info("cuCtxDestroy", &_destroy);
	print_info("cuCtxSetCurrent", &_current);
	print_info("cuStreamGetCtx", &_streamctx);
}

int main(void)
{
	int devCount = 0;

	printf("Start Test...\n");
	CHECK(cuInit(0));
	CHECK(cuDeviceGetCount(&devCount));
	CHECK(devCount == 0);
	CHECK(cuDeviceGet(&device, 0));

	cal_time_execution(1000);

	return 0;
}











