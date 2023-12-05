#include <vector>

#include <cuda_runtime_api.h>
#include <cublasLt.h>

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

#include "sample_cublasLt_LtHSHgemmStridedBatchSimple.h"
#include "helpers.h"
#include "unistd.h"

int main(int argc, char **argv) {
    size_t c;
    const char *format = "hT:P:m:n:k:";
    size_t repeat = 1;
    int dataPattern = 0;
    int m = 1024;
    int n = 1024;
    int k = 1024;
    while ((c = getopt(argc, argv, format)) != EOF) {
        switch (c) {
            case 'T': {
			      repeat = atoi(optarg);
			      if (repeat <= 0) repeat = 1;
			      break;
		      }
            case 'P': {
			      dataPattern = atoi(optarg);
			      if (dataPattern < 0 || dataPattern > 2) dataPattern = 0;
			      break;
		      }
            case 'm': {
			      m = atoi(optarg);
			      if (m < 0) m = 1024;
			      break;
		      }
            case 'n': {
			      n = atoi(optarg);
			      if (n < 0) n = 1024;
			      break;
		      }
            case 'k': {
			      k = atoi(optarg);
			      if (k < 0) k = 1024;
			      break;
		      }

            case 'h':
            default: printf("%s\n", format); exit(0);
	}
    }

    TestBench<__half, __half, float> props(m, n, k, 1.0f, 0.0f, 4 * 1024 * 1024 * 2 * 64, 1, dataPattern);

    props.run([&props, &repeat] {
        LtHSHgemmStridedBatchSimple(props.ltHandle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    props.m,
                                    props.n,
                                    props.k,
                                    &props.alpha,
                                    props.Adev,
                                    props.m,
                                    props.m * props.k,
                                    props.Bdev,
                                    props.k,
                                    props.k * props.n,
                                    &props.beta,
                                    props.Cdev,
                                    props.m,
                                    props.m * props.n,
                                    props.N,
                                    props.workspace,
                                    props.workspaceSize,
				    repeat);
    });

    return 0;
}
