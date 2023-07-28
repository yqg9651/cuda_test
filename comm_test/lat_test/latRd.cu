#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "compMalloc.h"
#include "helper_cuda.h"

struct mem_test {
    char *raw_addr;
    char *base;
    uint64_t *dummy;
    size_t raw_size;
    size_t len;
    size_t range;
    size_t stride;
    size_t page_size;
    bool compress_en;
    size_t repeat;
    unsigned warmup_cnt;
    FILE *fp;
};

// #define DRY_RUN
#ifdef DRY_RUN
#define SKIP_IF_DRY_RUN if (0)
#else
#define SKIP_IF_DRY_RUN
#endif

// static const size_t begin_bytes = 75665408;
// static const size_t begin_bytes = 67108864;
static const size_t begin_bytes = (8UL << 10);
static const size_t begin_stride = (32);

static void runLatRdTest(struct mem_test *mem);
static void runLatConstRdTest(struct mem_test *mem);
static void allocTestBuffer(struct mem_test *mem);
static void freeTestBuffer(struct mem_test *mem);
static size_t getGpuPageSize(void);
static void fileLineBegin(struct mem_test *mem);
static void fileLineEnd(struct mem_test *mem);
static void fileHeadInit(struct mem_test *mem);
static void runFineGrainedLatRdTest(struct mem_test *mem, bool reverse);

static size_t forward_stride(size_t stride)
{
    return stride * 2;
}

static size_t forward(size_t range)
{
    if (range < (32 << 10)) {
        range *= 2;
    } else if (range < (256 << 10)) {
        range += (32 << 10);
    } else if (range < (2UL << 20)) {
        range *= 2;
    } else if (range < (64UL << 20)) {
        range += (2UL << 20);
    } else if (range < (256UL << 20)) {
        range += (32UL << 20);
    } else if (range < (1UL << 30)) {
        range *= 2;
    } else if (range < (32UL << 30)) {
        // range += (1UL << 30);
        range *= 2;
    } else {
        size_t s;
	for (s = (1 << 30); s <= range; s *= 2) ;
	range += (s / 4);
    }

    return range;
}

static void init_repeat(struct mem_test *mem, size_t repeat)
{
    size_t c = (mem->range / mem->stride);
    if (c < (4 << 10)) {
        repeat *= 200;
    } else if (c < (64 << 10)) {
        repeat *= 80;
    } else if (c < (128 << 10)) {
        repeat *= 60;
    } else if (c < (1 << 20)) {
        repeat *= 50;
    } else if (c < (2 << 20)) {
        repeat *= 40;
    }

    if (mem->range > (1UL << 30)) {
        repeat = repeat / 10 + 1;
    } else if (mem->range > (2UL << 30)) {
        repeat = repeat / 40 + 1;
    }

    mem->repeat = repeat;
}

int main(int argc, char **argv)
{
    char *file_name = NULL;
    FILE *fp = stdout;
    size_t len = 2;
    size_t repeat = 100;
    size_t stride = 16;
    unsigned warmup_cnt = 0;
    bool compress_en = false;
    bool is_const = false;
    bool is_fine_grained = false;
    bool reverse = false;
    size_t c;
    const char *help_info = "[-F <output_file>] [-L <len(B)>] [-N <repeats>] [-C] [-S <stride_bytes>] [-h] [-R] [-G <fine_grained, p_chase>] [-W <warmup loop num>] [-X]";

    while ((c = getopt(argc, argv, "hCRXF:L:N:S:G:W:")) != EOF) {
        switch (c) {
            case 'F': {
                file_name = optarg;
		if ((fp = fopen(file_name, "w+")) == NULL) {
		    fprintf(stderr, "open file %s failed\n", file_name);
		    exit(-1);
		}

		break;
            }
            case 'N': {
                repeat = atoi(optarg);
		if (repeat <= 0) repeat = 100;
		break;
            }
            case 'L': {
                len = atol(optarg);
		if (len <= 0) len = 32;
		break;
            }
            case 'S': {
                stride = atoi(optarg);
		if (stride <= 0) stride = 16;
		break;
            }
            case 'C': {
                compress_en = true;
		break;
            }
            case 'R': {
                is_const = true;
		break;
            }
            case 'G': {
                if (strcmp("fine_grained", optarg) == 0) {
                    is_fine_grained = true;
                }

                break;
            }
            case 'W': {
                warmup_cnt = atoi(optarg);
                break;
            }
            case 'X': {
                reverse = true;
                break;
            }
            case 'h': {
            default:
		printf("%s\n", help_info);
		exit(-1);
            }
	}
    }

    printf("output file %s repeat %ld stride %ld compress %s attribute %s len %ldB warmup %d iter %ld %s (via %s)\n",
           (file_name ? file_name : "stdout"),
           repeat,
           (size_t)stride,
           (compress_en ? "enable" : "disable"),
           (is_const ? "const" : "normal"),
           (size_t)len,
           warmup_cnt,
           len / stride,
           (reverse ? "reverse order" : "order"),
           (is_fine_grained ? "fine-grained" : "p-chase"));

    struct mem_test mem = {0};

    cudaSetDevice(0);
    cudaDeviceReset();

    if (is_fine_grained == true) {
        mem.page_size = getGpuPageSize();
        mem.len = len;
        mem.range = len;
        mem.stride = stride;
        mem.repeat = repeat;
        mem.compress_en = compress_en;
        mem.fp = fp;
        mem.warmup_cnt = warmup_cnt;

        allocTestBuffer(&mem);
        runFineGrainedLatRdTest(&mem, reverse);
        return 0;
    }

    size_t range;
    size_t test_stride;

    mem.len = len;
    // mem.stride = stride;
    // mem.repeat = repeat;
    mem.page_size = getGpuPageSize();
    mem.compress_en = compress_en;
    mem.fp = fp;

SKIP_IF_DRY_RUN
    allocTestBuffer(&mem);

    fileHeadInit(&mem);

    for (test_stride = begin_stride; test_stride <= stride; test_stride = forward_stride(test_stride)) {
        mem.stride = test_stride;
        fileLineBegin(&mem);
        for (range = begin_bytes; range <= len; range = forward(range)) {
            mem.range = range;
            init_repeat(&mem, repeat);
SKIP_IF_DRY_RUN
            if (is_const == false) {
                if (mem.stride > mem.range) {
                    fprintf(mem.fp, ",0");
                    continue;
                }

                runLatRdTest(&mem);
            } else {
                if (mem.stride > mem.range) {
                    fprintf(mem.fp, ",0");
                    continue;
                }

                runLatConstRdTest(&mem);
            }

        }

        fileLineEnd(&mem);
    }

    freeTestBuffer(&mem);

    if (file_name) {
        fclose(fp);
    }

    return 0;
}

static size_t getGpuPageSize(void)
{
    return (4 * 1024);
}

static void fileLineBegin(struct mem_test *mem)
{
    char record[64];
    snprintf(record, sizeof(record), "%ld", mem->stride);
    fwrite(record, strlen(record), 1, mem->fp);
    fflush(mem->fp);
}

static void fileLineEnd(struct mem_test *mem)
{
    char record[64];
    snprintf(record, sizeof(record), "\n");
    fwrite(record, strlen(record), 1, mem->fp);
    fflush(mem->fp);
}

#define BUG_ON(expr) \
do { \
    bool b = !!(expr); \
    if (b == true) { \
        printf("BUG ON: %s\n", #expr); \
        exit(-1); \
    } \
} while (0)

static void fileHeadInit(struct mem_test *mem)
{
    size_t len = mem->len;
    size_t range;

    fprintf(mem->fp, "stride(Bytes)");
    for (range = begin_bytes; range <= len; range = forward(range)) {
        if (range < (1UL << 10)) {
            fprintf(mem->fp, ",%ldB", range);
        } else if (range < (1UL << 20)) {
            fprintf(mem->fp, ",%ldKB", range / 1024);
            BUG_ON(range % 1024 != 0);
        } else if (range < (1UL << 30)) {
            fprintf(mem->fp, ",%ldMiB", range / 1024 / 1024);
            BUG_ON(range % (1024 * 1024) != 0);
        } else {
            fprintf(mem->fp, ",%ldGB", range / 1024 / 1024 / 1024);
            BUG_ON(range % (1024 * 1024 * 1024) != 0);
        }
    }
    fprintf(mem->fp, "\n");
    fflush(mem->fp);
}

__global__ void fine_grained_latRdKernel(size_t iter, size_t test_num, unsigned warmup_cnt, void *start, size_t stride, uint64_t *dummy, uint32_t *cycles_out)
{
    extern __shared__ uint32_t s_cycles[];
    extern __shared__ uint32_t s_addrs[]; // the array is same as @s_cycles
    register char **p = (char **)start;
    register char *addr = (char *)start;
    register long long i;
    register uint32_t start_cycle;
    register uint32_t end_cycle;

    if (threadIdx.x != 0 || threadIdx.y != 0|| threadIdx.z != 0 || blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0) {
        return;
    }

    for (i = 0; i < test_num; ++i) {
        s_cycles[i] = 0;
        s_addrs[i] = 0;
    }

if (1) {
    // write warmup
    if (!warmup_cnt) {
    const size_t im_stride = 32; // must large than 0
    for (size_t ii = im_stride; ii < test_num * stride; ii += im_stride) {
        if (ii % stride == 0) {
            *(uint32_t *)&addr[ii - stride] = (uint32_t)((uint64_t)(&addr[ii]));
            // *(uint32_t *)&addr[ii - stride + 4] = (uint32_t)(((uint64_t)(&addr[ii])) >> 32);
            // *(uint64_t *)&addr[ii - stride] = (uint64_t)((uint64_t)(&addr[ii]));
            // *(uint64_t *)&addr[ii - stride + 8] = (uint64_t)((uint64_t)(&addr[ii + 8]));
            // *(uint64_t *)&addr[ii - stride + 16] = (uint64_t)((uint64_t)(&addr[ii + 16]));
            // *(uint64_t *)&addr[ii - stride + 24] = (uint64_t)((uint64_t)(&addr[ii + 24]));
            continue;
        }

        *(uint32_t *)&addr[ii - stride] = (uint32_t)((uint64_t)(&addr[ii]));
        // *(uint64_t *)&addr[ii - stride] = (uint64_t)((uint64_t)(&addr[ii]));
        // *(char **)&addr[ii] = (char *)&addr[ii];
    }

    // *(uint64_t *)&addr[test_num * stride - stride] = (uint64_t)((uint64_t)(&addr[0]));
    *(uint32_t *)&addr[test_num * stride - stride] = (uint32_t)((uint64_t)(&addr[0]));
    }
}

    for (i = 0L - (iter * warmup_cnt); i < (long)test_num; ++i) {
        if (i < 0) {
            p = (char **)*p;
            s_addrs[0] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
            continue;
        }

        start_cycle = clock();
        p = (char **)*p;
        // guarantee load finished
        s_addrs[i] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
        end_cycle = clock();

        s_cycles[i] = end_cycle - start_cycle;
    }

    memcpy(cycles_out, s_cycles, sizeof(uint32_t) * test_num);
}

#define TWICE(t) t t
#define T_1 p = (char **)*p;
#define T_2 TWICE(T_1)
#define T_4 TWICE(T_2)
#define T_8 TWICE(T_4)
#define T_16 TWICE(T_8)
#define T_32 TWICE(T_16)
#define T_64 TWICE(T_32)
#define T_128 TWICE(T_64)

__global__ void latRdKernel(size_t iter, void *start, size_t range, size_t stride, uint64_t *dummy)
{
    register char **p = (char **)start;
    register size_t i;
    register size_t cnt = range / (stride * 128) + 1;
    register size_t loop = iter;

    while (loop-- > 0) {
        for (i = 0; i < cnt; ++i) {
            T_128;
	}
    }

    (*dummy) += (uint64_t)p;
}

#define TC_1 p = (const char **)*p;
#define TC_2 TWICE(TC_1)
#define TC_4 TWICE(TC_2)
#define TC_8 TWICE(TC_4)
#define TC_16 TWICE(TC_8)
#define TC_32 TWICE(TC_16)
#define TC_64 TWICE(TC_32)
#define TC_128 TWICE(TC_64)

__global__ void latConstRdKernel(size_t iter, const void *start, int range, int stride, uint64_t *dummy)
{
    register const char **p = (const char **)start;
    register size_t i;
    register size_t cnt = range / (stride * 128) + 1;
    register size_t loop = iter;

    while (loop-- > 0) {
        for (i = 0; i < cnt; ++i) {
            TC_128;
	}
    }

    (*dummy) += (uint64_t)p;
}

__global__ void initStrideKernel(char *addr, size_t stride, size_t range)
{
    size_t i;

    for (i = stride; i < range; i += stride) {
        *(char **)&addr[i - stride] = (char *)&addr[i];
    }
    *(char **)&addr[i - stride] = (char *)&addr[0];
}

__global__ void initReverseStrideKernel(char *addr, size_t stride, size_t range)
{
    size_t i;

    for (i = range - stride; i > 0; i -= stride) {
        *(char **)&addr[i] = (char *)&addr[i - stride];
    }
    *(char **)&addr[0] = (char *)&addr[range - stride];
}

static unsigned long alignedUp(unsigned long old, unsigned long aligned_size)
{
    return ((old + aligned_size - 1) / aligned_size * aligned_size);
}

#define ALIGNED_UP(old, a) \
	alignedUp((unsigned long)(old), (unsigned long)(a))

static void allocTestBuffer(struct mem_test *mem)
{
    // size_t align_size = mem->page_size;
    size_t align_size = (2UL << 30);

    // init cuda
    checkCudaErrors(cudaFree(0));

    // mem->raw_size = mem->len + align_size;
    mem->raw_size = ((mem->len - 1) / align_size + 1) * align_size;
    checkCudaErrors(allocateCompressible((void **)&mem->raw_addr, mem->raw_size, mem->compress_en));
    // mem->base = (char *)ALIGNED_UP(mem->raw_addr, align_size);
    mem->base = (char *)ALIGNED_UP(mem->raw_addr, (2UL << 20));
    checkCudaErrors(cudaMalloc((void **)&mem->dummy, sizeof(*mem->dummy)));
    checkCudaErrors(cudaMemset(mem->raw_addr, 0, mem->raw_size));
}

static void freeTestBuffer(struct mem_test *mem)
{
    if (mem->raw_addr) {
        freeCompressible(mem->raw_addr, mem->raw_size, mem->compress_en);
	mem->raw_addr = NULL;
	mem->raw_size = 0;
    }

    if (mem->dummy) {
        checkCudaErrors(cudaFree((void *)mem->dummy));
	mem->dummy = NULL;
    }
}

static void initStride(struct mem_test *mem, bool reverse)
{
    dim3 threads, blocks; 
    threads = dim3(1, 1, 1);
    blocks  = dim3(1, 1, 1);

    if (reverse == false) {
        initStrideKernel<<<blocks, threads>>>(mem->base, mem->stride, mem->range);
    } else {
        initReverseStrideKernel<<<blocks, threads>>>(mem->base, mem->stride, mem->range);
    }
    checkCudaErrors(cudaDeviceSynchronize());
}

static inline double ts_sub_us(struct timespec *sts, struct timespec *ets)
{
    return (double)(ets->tv_sec - sts->tv_sec) * 1000.0 * 1000.0 + (double)(ets->tv_nsec - sts->tv_sec) / 1000.0;
}

static void runLatRd(struct mem_test *mem)
{
    cudaStream_t stream;
    dim3 threads, blocks;
    struct timespec sts, ets;
    char record[64];

    threads = dim3(1, 1, 1);
    blocks  = dim3(1, 1, 1);

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // warmup
    latRdKernel<<<blocks, threads, 0, stream>>>(1, mem->base, mem->range, mem->stride, mem->dummy);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // test
    clock_gettime(CLOCK_MONOTONIC_RAW, &sts);
    latRdKernel<<<blocks, threads, 0, stream>>>(mem->repeat, mem->base, mem->range, mem->stride, mem->dummy);
    checkCudaErrors(cudaStreamSynchronize(stream));
    clock_gettime(CLOCK_MONOTONIC_RAW, &ets);

    double us = ts_sub_us(&sts, &ets);
    size_t cnt = mem->range / (mem->stride * 128) + 1;
    cnt *= 128;
    cnt *= mem->repeat;

    snprintf(record, sizeof(record), ",%lf", us / (double)cnt * 1000.0);
    fwrite(record, strlen(record), 1, mem->fp);
    fflush(mem->fp);
    checkCudaErrors(cudaStreamDestroy(stream));
    printf("range %ld stride %ld mean %lf ns\n", mem->range, mem->stride, us / (double)cnt * 1000.0);
    fflush(stdout);
}

static void runLatRdTest(struct mem_test *mem)
{
    initStride(mem, false);
    runLatRd(mem);
}

static void runLatConstRd(struct mem_test *mem)
{
    cudaStream_t stream;
    dim3 threads, blocks;
    struct timespec sts, ets;
    char record[64];

    threads = dim3(1, 1, 1);
    blocks  = dim3(1, 1, 1);

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // warmup
    latConstRdKernel<<<blocks, threads, 0, stream>>>(1, mem->base, mem->range, mem->stride, mem->dummy);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // test
    clock_gettime(CLOCK_MONOTONIC_RAW, &sts);
    latConstRdKernel<<<blocks, threads, 0, stream>>>(mem->repeat, mem->base, mem->range, mem->stride, mem->dummy);
    checkCudaErrors(cudaStreamSynchronize(stream));
    clock_gettime(CLOCK_MONOTONIC_RAW, &ets);

    double us = ts_sub_us(&sts, &ets);
    size_t cnt = mem->range / (mem->stride * 128) + 1;
    cnt *= 128;
    cnt *= mem->repeat;

    snprintf(record, sizeof(record), ",%lf", us / cnt * 1000.0);
    fwrite(record, strlen(record), 1, mem->fp);
    fflush(mem->fp);
    checkCudaErrors(cudaStreamDestroy(stream));
    printf("range %ld stride %ld mean %lf ns\n", mem->range, mem->stride, us / cnt * 1000.0);
    fflush(stdout);
}

static void runLatConstRdTest(struct mem_test *mem)
{
    initStride(mem, false);
    runLatConstRd(mem);
}

static void runFineGrainedLatRd(struct mem_test *mem, bool reverse)
{
    cudaStream_t stream;
    dim3 threads, blocks;
    size_t s_size = 0; // share memory size
    void *host_outputs;
    void *dev_outputs;
    size_t iter = 0;

    BUG_ON((mem->stride > mem->range) || (mem->range / mem->stride == 0));

    int block_size = 0;
    int min_grid_size = 0;
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)fine_grained_latRdKernel));
    size_t dy_sm_size = 0;
    checkCudaErrors(cudaOccupancyAvailableDynamicSMemPerBlock(&dy_sm_size, (const void *)fine_grained_latRdKernel, 1, block_size));
    printf("block size %d grid size %d shared memory size %ld\n", block_size, min_grid_size, dy_sm_size);

    threads = dim3(block_size, 1, 1);
    blocks  = dim3(1, 1, 1);

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    iter = mem->range / mem->stride;
    s_size = dy_sm_size;
    size_t test_num = s_size / sizeof(uint32_t);
    if (iter < test_num) test_num = iter;
    // BUG_ON((iter * sizeof(uint32_t)) > s_size);
    // s_size = iter * sizeof(uint32_t);
    checkCudaErrors(cudaMalloc(&dev_outputs, s_size));
    checkCudaErrors(cudaMemset(dev_outputs, 0, s_size));
    host_outputs = calloc(s_size, 1);

    // test
    fine_grained_latRdKernel<<<blocks, threads, s_size, stream>>>(iter, test_num, mem->warmup_cnt, mem->base, mem->stride, mem->dummy, (uint32_t *)dev_outputs);
    checkCudaErrors(cudaMemcpyAsync(host_outputs, dev_outputs, s_size, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // record data to file
    {
    fprintf(mem->fp, "range %ldB stride %ldB share memory %ldB\n", mem->range, mem->stride, s_size);
    fprintf(mem->fp, "%15s,%20s,\t%-20s\n", "offset", "latency(cycles)", "addrs(l32)");
    uint32_t *lat = (uint32_t *)host_outputs;
    uint32_t min = ~0;
    uint32_t max = 0;
    double mean = 0;
    char *addr = (char *)mem->base;
    for (unsigned i = 0; i < test_num; ++i) {
        if (reverse == false) {
            fprintf(mem->fp, "%15ld,%20d,\t%p\n", i * mem->stride, lat[i], addr + i * mem->stride);
        } else {
            fprintf(mem->fp, "%15ld,%20d,\t%p\n", i * mem->stride, lat[i], i == 0 ? addr : addr + mem->range - i * mem->stride);
        }
        if (lat[i] < min) min = lat[i];
        if (lat[i] > max) max = lat[i];
        mean += (double)lat[i];
    }
    fflush(mem->fp);
    printf("latency max %d min %d mean %lf\n", max, min, mean / test_num);
    }

    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(dev_outputs));
    free(host_outputs);
}

static void runFineGrainedLatRdTest(struct mem_test *mem, bool reverse)
{
    initStride(mem, reverse);
    runFineGrainedLatRd(mem, reverse);
}
