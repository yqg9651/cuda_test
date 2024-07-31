#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "compMalloc.h"
#include "helper_cuda.h"

// FIXME allocate dynamically
#define MAX_DEV_NUM (8)

struct mem_test {
    char *raw_uc_ptrs[MAX_DEV_NUM];
    char *raw_mc_ptrs[MAX_DEV_NUM];
    char *uc_base[MAX_DEV_NUM];
    char *mc_base[MAX_DEV_NUM];
    uint64_t *dummy[MAX_DEV_NUM];
    CUmemGenericAllocationHandle ucHandle[MAX_DEV_NUM];

    CUmemGenericAllocationHandle mcHandle;
    size_t raw_size;
    size_t len;
    size_t range;
    size_t stride;
    size_t page_size;
    bool compress_en;
    size_t repeat;
    unsigned warmup_cnt;
    FILE *fp;
    int device_num;
    int requester_id;
    int data_init_id;
};

static inline size_t aligned_size(size_t size, size_t align)
{
    return ((size - 1) / align + 1) * align;
}

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
static void cudaInitHelper(struct mem_test *mem);

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
    int device_num = 1;
    const char *help_info = "[-F <output_file>] [-L <len(B)>] [-N <repeats>] [-C] [-S <stride_bytes>] [-h] [-R] [-G <fine_grained, p_chase>] [-W <warmup loop num>] [-X] [-D <device number>]";

    while ((c = getopt(argc, argv, "hCRXF:L:N:S:G:W:D:")) != EOF) {
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
            case 'D': {
                device_num = atoi(optarg);
                if (device_num <= 0) {
                    fprintf(stderr, "device number should be greater than 0(input %d)", device_num);
                    exit(-1);
                }
                break;
            }
            case 'h': {
            default:
		printf("%s\n", help_info);
		exit(-1);
            }
	}
    }

    printf("output file %s device num %d repeat %ld stride %ld compress %s attribute %s len %ldB warmup %d iter %ld %s (via %s)\n",
           (file_name ? file_name : "stdout"),
           device_num,
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

    if (is_fine_grained == true) {
        mem.page_size = getGpuPageSize();
        mem.len = len;
        mem.range = len;
        mem.stride = stride;
        mem.repeat = repeat;
        mem.compress_en = compress_en;
        mem.fp = fp;
        mem.warmup_cnt = warmup_cnt;
        mem.device_num = device_num;

        cudaInitHelper(&mem);
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
    mem.device_num = device_num;

    cudaInitHelper(&mem);

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

__global__ void loadDataKernel(void *dst, void *src, size_t size)
{
	memcpy(dst, src, size);
}

__global__ void fine_grained_latRdKernel(size_t iter, size_t test_num, unsigned warmup_cnt, void *start, size_t stride, uint64_t *dummy, uint32_t *cycles_out)
{
    extern __shared__ uint32_t s_cycles[];
    extern __shared__ uint32_t s_addrs[]; // the array is same as @s_cycles
    register char **p = (char **)__cvta_generic_to_global(start);
    register char *addr = (char *)__cvta_generic_to_global(start);
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

    // fine grained forward detect
// if (1) {
//     for (i = 0L; i < (long)2; ++i) {
//        start_cycle = clock();
// #if 0
//         p = (char **)*p;
// #else
//         asm("multimem.ld_reduce.relaxed.sys.global.add.u64 %0, [%1];" \
//             : "=l"(p) \
//             : "l"(p));
// #endif
//         // guarantee load finished
//         s_addrs[i] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
//         end_cycle = clock();
// 
//         s_cycles[i] = end_cycle - start_cycle;
// 	__threadfence_system();
// 	p = (char **)__cvta_generic_to_global((char *)start + stride);
// 	__threadfence_system();
// 	// __nanosleep(200 * 1000);
//     }
// 
//     memcpy(cycles_out, s_cycles, sizeof(uint32_t) * 2);
//     return;
// }

if (0) {
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
	// printf("xxxx %p\n", p);
        if (i < 0) {
#if 0
            p = (char **)*p;
#else
            asm("multimem.ld_reduce.relaxed.sys.global.add.u64 %0, [%1];" \
                : "=l"(p) \
                : "l"(p));
#endif
            s_addrs[0] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
            continue;
        }

        start_cycle = clock();
#if 0
        p = (char **)*p;
#else
        asm("multimem.ld_reduce.relaxed.sys.global.add.u64 %0, [%1];" \
            : "=l"(p) \
            : "l"(p));
#endif
        // guarantee load finished
        s_addrs[i] = static_cast<uint32_t>(reinterpret_cast<uint64_t>(p));
        end_cycle = clock();

        s_cycles[i] = end_cycle - start_cycle;
	// __nanosleep(200 * 1000);
    }

    memcpy(cycles_out, s_cycles, sizeof(uint32_t) * test_num);
}

#define TWICE(t) t t
#if 0
#define T_1 p = (char **)*p;
#else
#define T_1 \
        asm("multimem.ld_reduce.relaxed.sys.global.add.u64 %0, [%1];" \
            : "=l"(p) \
            : "l"(p));
#endif

#define T_2 TWICE(T_1)
#define T_4 TWICE(T_2)
#define T_8 TWICE(T_4)
#define T_16 TWICE(T_8)
#define T_32 TWICE(T_16)
#define T_64 TWICE(T_32)
#define T_128 TWICE(T_64)

__global__ void latRdKernel(size_t iter, void *start, size_t range, size_t stride, uint64_t *dummy)
{
    register char **p = (char **)__cvta_generic_to_global(start);
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

#if 0
#define TC_1 p = (const char **)*p;
#else
#define TC_1 \
        asm("multimem.ld_reduce.relaxed.sys.global.add.u64 %0, [%1];" \
            : "=l"(p) \
            : "l"(p));
#endif

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

__global__ void initStrideKernel(char *addr, size_t stride, size_t range, char *mc_addr)
{
    size_t i;

    for (i = stride; i < range; i += stride) {
        *(char **)&addr[i - stride] = (char *)&mc_addr[i];
    }
    *(char **)&addr[i - stride] = (char *)&mc_addr[0];
}

__global__ void initReverseStrideKernel(char *addr, size_t stride, size_t range, char *mc_addr)
{
    size_t i;

    for (i = range - stride; i > 0; i -= stride) {
        *(char **)&addr[i] = (char *)&mc_addr[i - stride];
    }
    *(char **)&addr[0] = (char *)&mc_addr[range - stride];
}

static unsigned long alignedUp(unsigned long old, unsigned long aligned_size)
{
    return ((old + aligned_size - 1) / aligned_size * aligned_size);
}

#define ALIGNED_UP(old, a) \
	alignedUp((unsigned long)(old), (unsigned long)(a))

static void allocTestBuffer(struct mem_test *mem)
{
    // init cuda
    cudaInitHelper(mem);


#if 0
    // size_t align_size = mem->page_size;
    size_t align_size = (2UL << 30);

    // mem->raw_size = mem->len + align_size;
    mem->raw_size = ((mem->len - 1) / align_size + 1) * align_size;
    checkCudaErrors(allocateCompressible((void **)&mem->raw_uc_ptrs, mem->raw_size, mem->compress_en));
    // mem->uc_base[0] = (char *)ALIGNED_UP(mem->raw_uc_ptrs, align_size);
    mem->uc_base[0] = (char *)ALIGNED_UP(mem->raw_uc_ptrs, (2UL << 20));
    checkCudaErrors(cudaMalloc((void **)&mem->dummy, sizeof(*mem->dummy)));
    checkCudaErrors(cudaMemset(mem->raw_uc_ptrs, 0, mem->raw_size));
#endif

    // TODO
    // create mc handler via device 0
    checkCudaErrors(cudaSetDevice(0));

    CUmulticastObjectProp mcprop;
    memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
    mcprop.numDevices = mem->device_num;
    // mcprop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    mcprop.flags = 0;
    size_t ucgran, mcgran;
    CUmulticastGranularity_flags mcOption = CU_MULTICAST_GRANULARITY_RECOMMENDED;
    checkCudaErrors(cuMulticastGetGranularity(&mcgran, &mcprop, mcOption));

    mem->raw_size = aligned_size(mem->len, mcgran);
    mcprop.size = mem->raw_size;
    checkCudaErrors(cuMulticastCreate(&mem->mcHandle, &mcprop));
    // char shareableHandle[64];
    // chekcCudaErrors(cuMemExportToShareableHandle(shareableHandle, *mcHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0))

    // map mc handler
    for (int i = 0; i < mem->device_num; ++i) {
        CUdevice dev;
        checkCudaErrors(cuDeviceGet(&dev, i));

        checkCudaErrors(cudaSetDevice(i));
        // CUmemGenericAllocationHandle mcHandle;
        // checkCudaErrors(cuMemImportFromShareableHandle(&mcHandle, (void *)shareableHandle, CU_MEM_HANDLE_TYPE_FABRIC));

        checkCudaErrors(cuMulticastAddDevice(mem->mcHandle, dev));
    }

    for (int i = 0; i < mem->device_num; ++i) {
        CUdevice dev;
        checkCudaErrors(cuDeviceGet(&dev, i));

        checkCudaErrors(cudaSetDevice(i));

	CUmemAllocationProp ucprop;
        memset(&ucprop, 0, sizeof(CUmemAllocationProp));
        ucprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ucprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ucprop.location.id = dev;
        // ucprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        if (mem->compress_en == true) {
            ucprop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        }

        checkCudaErrors(cuMemGetAllocationGranularity(&ucgran, &ucprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        checkCudaErrors(cuMemAddressReserve((CUdeviceptr*)&mem->raw_uc_ptrs[i], mem->raw_size, ucgran, 0U, 0));
        checkCudaErrors(cuMemCreate(&mem->ucHandle[i], mem->raw_size, &ucprop, 0));
        checkCudaErrors(cuMemMap((CUdeviceptr)mem->raw_uc_ptrs[i], mem->raw_size, 0, mem->ucHandle[i], 0));
        CUmemAccessDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = dev;
        checkCudaErrors(cuMemSetAccess((CUdeviceptr)mem->raw_uc_ptrs[i], mem->raw_size, &desc, 1));
        checkCudaErrors(cudaMemset(mem->raw_uc_ptrs[i], 0, mem->raw_size));

        checkCudaErrors(cuMulticastBindMem(mem->mcHandle, 0/*mcOffset*/, mem->ucHandle[i], 0/*memOffset*/, mem->raw_size, 0/*flags*/));

        // map mc va
        checkCudaErrors(cuMemAddressReserve((CUdeviceptr*)&mem->raw_mc_ptrs[i], mem->raw_size, mcgran, 0U, 0));
        checkCudaErrors(cuMemMap((CUdeviceptr)mem->raw_mc_ptrs[i], mem->raw_size, 0, mem->mcHandle, 0));
        checkCudaErrors(cuMemSetAccess((CUdeviceptr)mem->raw_mc_ptrs[i], mem->raw_size, &desc, 1));

        // base is raw_mc_ptrs
        mem->uc_base[i] = (char *)ALIGNED_UP(mem->raw_uc_ptrs[i], (2UL << 20));
        mem->mc_base[i] = (char *)ALIGNED_UP(mem->raw_mc_ptrs[i], (2UL << 20));
        checkCudaErrors(cudaMalloc((void **)&mem->dummy[i], sizeof(*mem->dummy[i])));
    }
}

static void freeTestBuffer(struct mem_test *mem)
{
    for (int i = 0; i < mem->device_num; ++i) {
        CUdevice dev;
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cuCtxGetDevice(&dev));
        checkCudaErrors(cuMulticastUnbind(mem->mcHandle, dev, 0/*mcOffset*/, mem->raw_size));

        // Release the UC memory and mapping
        checkCudaErrors(cuMemUnmap((CUdeviceptr)mem->raw_uc_ptrs[i], mem->raw_size));
        checkCudaErrors(cuMemAddressFree((CUdeviceptr)mem->raw_uc_ptrs[i], mem->raw_size));
        checkCudaErrors(cuMemRelease(mem->ucHandle[i]));

        // Release the MC memory and mapping
        checkCudaErrors(cuMemUnmap((CUdeviceptr)mem->raw_mc_ptrs[i], mem->raw_size));
        checkCudaErrors(cuMemAddressFree((CUdeviceptr)mem->raw_mc_ptrs[i], mem->raw_size));

        if (mem->dummy[i]) {
            checkCudaErrors(cudaFree((void *)mem->dummy[i]));
    	    mem->dummy[i] = NULL;
        }
    }
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cuMemRelease(mem->mcHandle));
}

static void initStride(struct mem_test *mem, bool reverse)
{
    dim3 threads, blocks; 
    threads = dim3(1, 1, 1);
    blocks  = dim3(1, 1, 1);

    int dev_id = 1;
    checkCudaErrors(cudaSetDevice(dev_id));
    if (reverse == false) {
        initStrideKernel<<<blocks, threads>>>(mem->uc_base[dev_id], mem->stride, mem->range, mem->mc_base[0]);
    } else {
        initReverseStrideKernel<<<blocks, threads>>>(mem->uc_base[dev_id], mem->stride, mem->range, mem->mc_base[0]);
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
    latRdKernel<<<blocks, threads, 0, stream>>>(1, mem->uc_base[0], mem->range, mem->stride, mem->dummy[0]);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // test
    clock_gettime(CLOCK_MONOTONIC_RAW, &sts);
    latRdKernel<<<blocks, threads, 0, stream>>>(mem->repeat, mem->mc_base[0], mem->range, mem->stride, mem->dummy[0]);
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
    checkCudaErrors(cudaSetDevice(0));
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
    latConstRdKernel<<<blocks, threads, 0, stream>>>(1, mem->uc_base[0], mem->range, mem->stride, mem->dummy[0]);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // test
    clock_gettime(CLOCK_MONOTONIC_RAW, &sts);
    latConstRdKernel<<<blocks, threads, 0, stream>>>(mem->repeat, mem->mc_base[0], mem->range, mem->stride, mem->dummy[0]);
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
    checkCudaErrors(cudaSetDevice(1));
    initStride(mem, false);
    checkCudaErrors(cudaSetDevice(0));
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

    // warmup L2 cache
    if (0) {
    char *addr;
    size_t l2_size = 120 << 20;
    checkCudaErrors(cudaMalloc(&addr, l2_size * 2));
    loadDataKernel<<<1, 1, 1, stream>>>(addr, addr + l2_size, l2_size);
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaFree(addr));
    }

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
    // FIXME test_num
    test_num = 2;
    fine_grained_latRdKernel<<<blocks, threads, s_size, stream>>>(iter, test_num, mem->warmup_cnt, mem->mc_base[0], mem->stride, mem->dummy[0], (uint32_t *)dev_outputs);
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
    char *addr = (char *)mem->uc_base[0];
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
    checkCudaErrors(cudaSetDevice(1));
    initStride(mem, false);
    checkCudaErrors(cudaSetDevice(0));
    runFineGrainedLatRd(mem, reverse);
}

static void cudaInitHelper(struct mem_test *mem)
{
    static bool init = false;
    static int done = 0;

    if (__sync_bool_compare_and_swap(&init, false, true) == false) {
	    while (__sync_fetch_and_add(&done, 0) == 0) {}
	    return;
    }

    for (int i = 0; i < mem->device_num; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaDeviceReset());
        checkCudaErrors(cudaFree(0));
    }

    done = 1;
}
