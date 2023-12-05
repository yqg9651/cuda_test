/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cublasLt.h>
#include <functional>

#include "helpers.h"
#include "sample_cublasLt_LtFp8Matmul.h"

__global__ void noopKernel(unsigned int cycles)
{
	unsigned int start = clock();
	volatile unsigned int *s = &start;
	unsigned int end = 0;
re:
	end = clock();
	if (end - *s < cycles) goto re;
}

/// Sample wrapper executing fp8 matmul with cublasLtMatmul, with addition of per-tensor scaling, amax calculations, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
void LtFp8Matmul(cublasLtHandle_t ltHandle,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *a_scale, /* device pointer */
                 const __nv_fp8_e4m3 *A,
                 int lda,
                 const float *b_scale, /* device pointer */
                 const __nv_fp8_e4m3 *B,
                 int ldb,
                 const float *c_scale, /* device pointer */
                 __nv_fp8_e4m3 *D,
                 int ldc,
                 const float *d_scale, /* device pointer */
                 float *amax_d, /* device pointer */
                 void *workspace,
                 size_t workspaceSize,
		 int repeats,
		 int8_t fast_mode) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    float beta = 0.0; // Can be non-zero starting from 12.0

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
    // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC, &transb, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &transa, sizeof(transa)));

    // for perf
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_mode, sizeof(fast_mode)));

    // set scaling factors
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    // checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, m, n, ldc));
    printf("%d %d %d %d\n", lda, ldb, ldc, ldc);

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    cudaStream_t stream = NULL;
    checkCudaStatus(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    checkCudaStatus(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    checkCudaStatus(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    for (int loop = 0; loop < repeats + 1; ++loop) {
	    if (loop == 1) checkCudaStatus(cudaEventRecord(startEvent, stream));
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     nullptr,
                                     Cdesc,
                                     D,
                                     Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     stream));
	    if (loop == 0) checkCudaStatus(cudaDeviceSynchronize());
    }
    checkCudaStatus(cudaEventRecord(stopEvent, stream));
    checkCudaStatus(cudaEventSynchronize(stopEvent));
    float ms;
    checkCudaStatus(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("%s: %lf TFlops(%d, %d, %d, %d, ms=%f)\n", __func__, (double)m * n * k * 2 / 1000.0 / 1000.0 * repeats / ms / 1000.0, m, n, k, (int)repeats, ms);
    if (startEvent) checkCudaStatus(cudaEventDestroy(startEvent));
    if (stopEvent) checkCudaStatus(cudaEventDestroy(stopEvent));
    if (stream) checkCudaStatus(cudaStreamDestroy(stream));

{
    std::function<bool(uint8_t)> is_nan_fp8_e4m3 = [](uint8_t v) -> bool {
        uint8_t exp = (v & (0xFU << 3)) >> 3;
	uint8_t frac = (v & 0x7U);
	return (exp == 0xFU) && (frac != 0);
    };
    std::function<bool(uint8_t)> is_inf_fp8_e4m3 = [](uint8_t v) -> bool {
        uint8_t exp = (v & (0xFU << 3)) >> 3;
	uint8_t frac = (v & 0x7U);
        return (exp == 0xFU) && (frac == 0);
    };
    std::function<bool(uint8_t)> is_zero_fp8_e4m3 = [](uint8_t v) -> bool {
        uint8_t exp = (v & (0xFU << 3)) >> 3;
	uint8_t frac = (v & 0x7U);
        return (exp == 0) && (frac == 0);
    };
    const size_t result_sz = sizeof(uint8_t) * m * n;
    uint8_t *temp_D = static_cast<uint8_t *>(std::calloc(1, result_sz));

    checkCudaStatus(cudaMemcpy(temp_D, D, result_sz, cudaMemcpyDeviceToHost));
    size_t is_nan = 0;
    size_t is_inf = 0;
    size_t is_zero = 0;
    for (size_t r = 0; r < m * n; ++r) {
        is_nan += is_nan_fp8_e4m3(temp_D[r]);
	is_inf += is_inf_fp8_e4m3(temp_D[r]);
	is_zero += is_zero_fp8_e4m3(temp_D[r]);
    }
#define P_100(v) ((double)v / (m * n))
    printf("\n %lf is NaN \n %lf is Inf \n %lf is Zero\n", P_100(is_nan), P_100(is_inf), P_100(is_zero));
    std::free(temp_D);
}
    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
