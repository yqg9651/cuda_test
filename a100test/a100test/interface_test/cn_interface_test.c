#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "math.h"
#include "cn_api.h"

#define CHECK(func)	\
do {	\
	int ret;	\
	if ((ret = (func)) != 0) {	\
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

CNdev device;

void cal_time_execution(int count)
{
	int i;
	CNcontext ctx1, ctx2, str_ctx;
	CNqueue stream;
	HOSTaddr d_a;
	struct time_pack _create, _destroy, _current, _streamctx, _atomic, _streamatomic;

	pack_init(count, &_create);
	pack_init(count, &_destroy);
	pack_init(count, &_current);
	pack_init(count, &_streamctx);
	pack_init(count, &_atomic);
	pack_init(count, &_streamatomic);

	for (i = 0; i < count; i++) {
		time_start(&_create);
		CHECK(cnCtxCreate(&ctx1, 0, device));
		time_end(&_create);

		CHECK(cnMallocHost(&d_a, 1024));

		CHECK(cnCtxCreate(&ctx2, 0, device));

		time_start(&_current);
		CHECK(cnCtxSetCurrent(ctx1));
		time_end(&_current);

		CHECK(cnCreateQueue(&stream, 0));
		CHECK(cnCtxSetCurrent(ctx2));

		time_start(&_streamctx);
		CHECK(cnQueueGetContext(stream, &str_ctx));
		time_end(&_streamctx);

		CHECK(str_ctx != ctx1);

		time_start(&_atomic);
		CHECK(cnAtomicOperation((uint64_t)d_a, 4, 0, CN_ATOMIC_OP_REQUEST,
				CN_ATOMIC_REQUEST_SET));
		//CHECK(cnAtomicOperation((uint64_t)d_a, 4, 0, CN_ATOMIC_OP_COMPARE,
		//		CN_ATOMIC_COMPARE_EQUAL));
		time_end(&_atomic);


		time_start(&_streamatomic);
		CHECK(cnQueueAtomicOperation(stream, (uint64_t)d_a, 4, 0,
				CN_ATOMIC_OP_REQUEST,
				CN_ATOMIC_REQUEST_SET));
		//CHECK(cnQueueAtomicOperation(stream, (uint64_t)d_a, 4, 0,
		//		CN_ATOMIC_OP_COMPARE,
		//		CN_ATOMIC_COMPARE_EQUAL));
		time_end(&_streamatomic);

		CHECK(cnQueueSync(stream));
		CHECK(cnDestroyQueue(stream));

		time_start(&_destroy);
		CHECK(cnCtxDestroy(ctx1));
		time_end(&_destroy);
		CHECK(cnCtxDestroy(ctx2));
	}


	print_info("cnCtxCreate", &_create);
	print_info("cnCtxDestroy", &_destroy);
	print_info("cnCtxSetCurrent", &_current);
	print_info("cnQueueGetContext", &_streamctx);
	print_info("cnAtomicOperation", &_atomic);
	print_info("cnQueueAtomicOperation", &_streamatomic);
}

int main(void)
{
	int devCount = 0;

	printf("Start Test...\n");
	CHECK(cnInit(0));
	CHECK(cnDeviceGetCount(&devCount));
	CHECK(devCount == 0);
	CHECK(cnDeviceGet(&device, 0));

	cal_time_execution(100);

	return 0;
}











