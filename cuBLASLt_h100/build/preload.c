#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>       
#include <dlfcn.h>        
#include <pthread.h>      
#include <string.h>       
#include <stdlib.h>       
#include <execinfo.h>     
#include "cuda.h"
#include "cublasLt.h"

// int cublasLtMatmulDescCreate(void* matmulDesc,
//                                                      int computeType,
//                                                     int scaleType);
extern void *_dl_sym(void *handle, const char *name, void *who);

cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType);
typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t* matmulDesc,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType);
static cublasLtMatmulDescCreate_t cublasLtMatmulDescCreate_raw;

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc,
                                                     cublasComputeType_t computeType,
                                                     cudaDataType_t scaleType)
{
	if (cublasLtMatmulDescCreate_raw == NULL) {
		cublasLtMatmulDescCreate_raw = (cublasLtMatmulDescCreate_t)_dl_sym(RTLD_NEXT, "cublasLtMatmulDescCreate", cublasLtMatmulDescCreate);
		if (cublasLtMatmulDescCreate_raw == NULL) {
			void *lib = dlopen("libcublasLt.so", RTLD_NOW);
			cublasLtMatmulDescCreate_raw = (cublasLtMatmulDescCreate_t)_dl_sym(lib, "cublasLtMatmulDescCreate", cublasLtMatmulDescCreate);
		}
	}

	if (cublasLtMatmulDescCreate_raw != NULL) printf("hook %s succcess\n", __func__);
	int rc = cublasLtMatmulDescCreate_raw(matmulDesc, computeType, scaleType);
	int8_t fast_mode = 1;
	cublasLtMatmulDescSetAttribute(*matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_mode, sizeof(fast_mode));
	return rc;
}
