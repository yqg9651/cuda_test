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

#include "sample_cublasLt_LtHSHgemmStridedBatchSimple.h"
#include "helpers.h"

/// Sample wrapper executing mixed precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasGemmEx,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed
void LtHSHgemmStridedBatchSimple(cublasLtHandle_t ltHandle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float *alpha, /* host pointer */
                                 const __half *A,
                                 int lda,
                                 int64_t stridea,
                                 const __half *B,
                                 int ldb,
                                 int64_t strideb,
                                 const float *beta, /* host pointer */
                                 __half *C,
                                 int ldc,
                                 int64_t stridec,
                                 int batchCount,
                                 void *workspace,
                                 size_t workspaceSize,
				 size_t repeat) {

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    printf("lda %d ldb %d ldc %d\n", lda, ldb, ldc);
    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
// #define USE_32F
#ifdef USE_32F
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
#else
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
#endif
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // create matrix descriptors, we need to configure batch size and counts in this case
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));

    // in this simplified example we take advantage of cublasLtMatmul shortcut notation with algo=NULL which will force
    // matmul to get the basic heuristic result internally. Downsides of this approach are that there is no way to
    // configure search preferences (e.g. disallow tensor operations or some reduction schemes) and no way to store the
    // algo for later use
    cublasLtMatmulPreference_t preference = nullptr;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
	    printf("warning: !!!! Unable to find any suitable algorithms\n");
    }

#if 0
    // cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t Border = CUBLASLT_ORDER_COL32_2R_4R4;
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &Border, sizeof(rowOrder)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
#endif

    cudaEvent_t startEvent = NULL, stopEvent = NULL;
    cudaStream_t stream = NULL;
    checkCudaStatus(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    checkCudaStatus(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    checkCudaStatus(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#ifdef USE_32F
#else
    __half alpha_fp16 = __float2half_rn(*alpha);
    __half beta_fp16 = __float2half_rn(*beta);
#endif
    for (int loop = 1; loop < repeat + 1; ++loop) {
    if (loop == 1) {
        checkCudaStatus(cudaEventRecord(startEvent, stream));
    }
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
#ifdef USE_32F
				     alpha,
#else
                                     &alpha_fp16,
#endif
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
#ifdef USE_32F
				     beta,
#else
                                     &beta_fp16,
#endif
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     stream));
    if (loop == 0) {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }
    }
    checkCudaStatus(cudaEventRecord(stopEvent, stream));
    checkCudaStatus(cudaEventSynchronize(stopEvent));
    float ms;
    checkCudaStatus(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("%s: %lf TFlops(%d, %d, %d, %d, %d, ms=%f)\n", __func__, (double)m * n * k * 2 * batchCount / 1000.0 / 1000.0 * repeat / ms / 1000.0, m, n, k, batchCount, (int)repeat, ms);
    if (startEvent) checkCudaStatus(cudaEventDestroy(startEvent));
    if (stopEvent) checkCudaStatus(cudaEventDestroy(stopEvent));
    if (stream) checkCudaStatus(cudaStreamDestroy(stream));

{
    std::function<bool(uint16_t)> is_nan_fp16 = [](uint16_t v) -> bool {
        uint16_t exp = (v & (0x1FU << 10)) >> 10;
	uint16_t mant = v & 0x3FFU;
	return (exp == 0x1FU) && (mant != 0);
    };

    std::function<bool(uint16_t)> is_inf_fp16 = [](uint16_t v) -> bool {
        uint16_t exp = (v & (0x1FU << 10)) >> 10;
	uint16_t mant = v & 0x3FFU;
	return (exp == 0x1FU) && (mant == 0);
    };

    std::function<bool(uint16_t)> is_zero_fp16 = [](uint16_t v) -> bool {
        uint16_t exp = (v & (0x1FU << 10)) >> 10;
	uint16_t mant = v & 0x3FFU;
	return (exp == 0) && (mant == 0);
    };

    const size_t result_sz = sizeof(uint16_t) * m * n;
    uint16_t *temp_C = static_cast<uint16_t *>(std::calloc(1, result_sz));

    checkCudaStatus(cudaMemcpy(temp_C, C, result_sz, cudaMemcpyDeviceToHost));
    size_t is_nan = 0;
    size_t is_inf = 0;
    size_t is_zero = 0;
    for (size_t r = 0; r < m * n; ++r) {
        is_nan += is_nan_fp16(temp_C[r]);
	is_inf += is_inf_fp16(temp_C[r]);
	is_zero += is_zero_fp16(temp_C[r]);
    }
#define P_100(v) ((double)v / (m * n))
    printf("\n %lf is NaN \n %lf is Inf \n %lf is Zero\n", P_100(is_nan), P_100(is_inf), P_100(is_zero));
    std::free(temp_C);

}

    // descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}
