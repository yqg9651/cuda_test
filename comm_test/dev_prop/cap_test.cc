#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "cuda.h"

#define CUDA_CHECK(exp) \
do { \
    CUresult ret = (exp); \
    if (ret != CUDA_SUCCESS) { \
        const char *str; \
        (void)cuGetErrorString(ret, &str); \
        fprintf(stderr, "%s@%d: call %s return %d(%s)\n", __func__, __LINE__, #exp, ret, str); \
        exit(-1); \
    } \
} while (0)

/* The helper macro list to device properties */
#define CUDA_DEV_ATTR_LIST(op) \
    op(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) \
    op(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X) \
    op(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y) \
    op(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z) \
    op(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X) \
    op(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y) \
    op(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z) \
    op(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) \
    op(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK) \
    op(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY) \
    op(CU_DEVICE_ATTRIBUTE_WARP_SIZE) \
    op(CU_DEVICE_ATTRIBUTE_MAX_PITCH) \
    op(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK) \
    op(CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK) \
    op(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) \
    op(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT) \
    op(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP) \
    op(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) \
    op(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT) \
    op(CU_DEVICE_ATTRIBUTE_INTEGRATED) \
    op(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY) \
    op(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES) \
    op(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT) \
    op(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS) \
    op(CU_DEVICE_ATTRIBUTE_ECC_ENABLED) \
    op(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID) \
    op(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID) \
    op(CU_DEVICE_ATTRIBUTE_TCC_DRIVER) \
    op(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) \
    op(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE) \
    op(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) \
    op(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT) \
    op(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS) \
    op(CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE) \
    op(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID) \
    op(CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT) \
    op(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) \
    op(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) \
    op(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH) \
    op(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR) \
    op(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR) \
    op(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY) \
    op(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD) \
    op(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID) \
    op(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO) \
    op(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS) \
    op(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) \
    op(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM) \
    op(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1) \
    op(CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1) \
    op(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1) \
    op(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) \
    op(CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH) \
    op(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) \
    op(CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES) \
    op(CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES) \
    op(CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST) \
    op(CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR) \
    op(CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE) \
    op(CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE) \
    op(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK) \
    op(CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS) \
    op(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING) \
    op(CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES) \
    op(CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH) \
    op(CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS) \
    op(CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR) \
    op(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT) \
    op(CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS) \
    op(CU_DEVICE_ATTRIBUTE_NUMA_CONFIG) \
    op(CU_DEVICE_ATTRIBUTE_NUMA_ID) \
    op(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED) \
    op(CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID)

static void cuda_dev_attr_dump(int ordinal)
{
    CUdevice dev;
    CUDA_CHECK(cuDeviceGet(&dev, ordinal));
    printf("\n***********************************%s %d****************************************\n", "Dump device attribute", (int)ordinal);
#define op(attr) \
do { \
    int val; \
    CUDA_CHECK(cuDeviceGetAttribute(&val, attr, dev)); \
    printf("%-70s %10d\n", #attr, val); \
} while (0);
    CUDA_DEV_ATTR_LIST(op);
#undef op
    printf("********************************************************************************\n\n");
}

int main(void)
{
    CUDA_CHECK(cuInit(0));

    int count;
    CUDA_CHECK(cuDeviceGetCount(&count));
    for (int i = 0; i < count; ++i) {
        cuda_dev_attr_dump(i);
    }

    return 0;
}
