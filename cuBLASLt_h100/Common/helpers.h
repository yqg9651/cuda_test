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

#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <functional>
#include <random>

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

template <typename InType, typename OutType = InType, typename ComputeType = OutType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k, ComputeType alpha = 0.0f, ComputeType beta = 0.0f, size_t workspaceSize = 1024 * 1024 * 4, int N = 1, int dataPattern = 2,
            ComputeType Ascale = 2.0, ComputeType Bscale = 0.5, ComputeType Cscale = 1.0, ComputeType Dscale = 1.0) :
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(m * k * N), Bhost(n * k * N),
        Chost(m * n * N), biasHost(m * N), AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale), dataPattern(dataPattern) {
        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(OutType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDev), m * N * sizeof(OutType)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        // Currently only fp8 supports per-tensor scaling
        perTensorScalingEnabled = std::is_same<InType, __nv_fp8_e4m3>::value || std::is_same<InType, __nv_fp8_e5m2>::value;

        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(*AscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(*BscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(*CscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(*DscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(*DamaxDev)));
        }

	switch (dataPattern) {
            case 0: clearData(); break;
            case 1: maxPowerData(); break;
            case 2: 
            default: randomData();
	}
    }

    ~TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(Cdev));
        checkCudaStatus(cudaFree(biasDev));
        checkCudaStatus(cudaFree(workspace));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaFree(AscaleDev));
            checkCudaStatus(cudaFree(BscaleDev));
            checkCudaStatus(cudaFree(CscaleDev));
            checkCudaStatus(cudaFree(DscaleDev));
            checkCudaStatus(cudaFree(DamaxDev));
        }
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    void maxPowerData() {
        for (int i = 0; i < m * k * N; i++) Ahost[i] = InType(i);
        for (int i = 0; i < n * k * N; i++) Bhost[i] = InType(i);
        for (int i = 0; i < m * N; i++) biasHost[i] = InType(i + 1);
    }

    void clearData() {
        for (int i = 0; i < m * k * N; i++) memset(&Ahost[i], 0, sizeof(Ahost[i]));
        for (int i = 0; i < n * k * N; i++) memset(&Bhost[i], 0, sizeof(Bhost[i]));
        for (int i = 0; i < m * N; i++) memset(&biasHost[i], 0, sizeof(biasHost[i]));
    }

    void randomData() {
        std::mt19937 engine(std::random_device{}());
	uint64_t max = 0;
	uint64_t shift = 0;
	switch (sizeof(InType)) {
            case 1: max = std::numeric_limits<uint8_t>::max(); shift = 56; break;
            case 2: max = std::numeric_limits<uint16_t>::max(); shift = 48; break;
            case 4: max = std::numeric_limits<uint32_t>::max(); shift = 32; break;
            case 8:
            default: max = std::numeric_limits<uint64_t>::max(); shift = 0;
	}
        std::uniform_int_distribution<uint64_t> dist(0, max);

	auto generate = [&]() -> long long int {
            uint64_t num = dist(engine);
	    num |= (num << shift);
	    return num;
	};

        for (int i = 0; i < m * k * N; i++) Ahost[i] = InType(generate());
        for (int i = 0; i < n * k * N; i++) Bhost[i] = InType(generate());
        for (int i = 0; i < m * N; i++) biasHost[i] = InType(generate());
    }

    void copyDataToDevice() {
        checkCudaStatus(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaMemcpyAsync(AscaleDev, &AscaleHost, sizeof(AscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(BscaleDev, &BscaleHost, sizeof(BscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(CscaleDev, &CscaleHost, sizeof(CscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DscaleDev, &DscaleHost, sizeof(DscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), cudaMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        checkCudaStatus(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();
        streamSynchronize();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    bool perTensorScalingEnabled;
    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<OutType> Chost, biasHost;
    void *workspace;
    InType *Adev, *Bdev;
    OutType *Cdev, *biasDev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
    ComputeType AscaleHost, BscaleHost, CscaleHost, DscaleHost, DamaxHost;
    ComputeType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DamaxDev;
    int dataPattern;
    const int mpAligned = 16;
};

template <>
inline void TestBench<__nv_fp8_e4m3, __nv_fp8_e4m3, float>::maxPowerData() {
    const uint8_t n1 = 0x5a;
    const uint8_t n2 = 0xa5;
    for (int batch = 0; batch < N; batch++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int pos = i * k + j;
		// int mpos = i + m * j + m * k * batch;
		int mpos = i * k + j + m * k * batch;
		if ((pos / (k)) % 2 == 0) memcpy(&Ahost[mpos], &n1, sizeof(Ahost[0]));
		else memcpy(&Ahost[mpos], &n2, sizeof(Ahost[0]));
	    }
	}

	for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int pos = i * k + j;
		int mpos = i * k + j + n * k * batch;
		if ((pos / (mpAligned * 2)) % 2 == 0) memcpy(&Bhost[mpos], &n2, sizeof(Bhost[0]));
		else memcpy(&Bhost[mpos], &n1, sizeof(Bhost[0]));
	    }
	}

    }
    for (int i = 0; i < m * N; i++) biasHost[i] = __nv_fp8_e4m3(i + 1);
}

template <>
inline void TestBench<__half, __half, float>::maxPowerData() {
    const uint16_t n1 = 0x5a5a;
    const uint16_t n2 = 0xa5a5;
    for (int batch = 0; batch < N; batch++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int pos = i * k + j;
		int mpos = i + m * j + m * k * batch;
		if ((pos / k) % 2 == 0) memcpy(&Ahost[mpos], &n1, sizeof(Ahost[0]));
		else memcpy(&Ahost[mpos], &n2, sizeof(Ahost[0]));
	    }
	}

	for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int pos = i * k + j;
		int mpos = i * k + j + n * k * batch;
		if ((pos / mpAligned) % 2 == 0) memcpy(&Bhost[mpos], &n2, sizeof(Bhost[0]));
		else memcpy(&Bhost[mpos], &n1, sizeof(Bhost[0]));
	    }
	}
#if 0
	if (batch != 0) continue;
	printf("matrix A\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
		    int mpos = i + m * j;
		    uint16_t fl = *((uint16_t *)&Ahost[mpos]);
		    printf("%#4x ", fl);
	    }
	    printf("\n");
	}
	printf("matrix B\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
		    int mpos = i + n * j;
		    uint16_t fl = *((uint16_t *)&Bhost[mpos]);
		    printf("%#4x ", fl);
	    }
	    printf("\n");
	}
#endif

    }
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__half, __half, cuComplex>::maxPowerData() {
    const uint16_t n1 = 0x5a5a;
    const uint16_t n2 = 0xa5a5;
    for (int batch = 0; batch < N; batch++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                int pos = i * k + j;
		int mpos = i + m * j + m * k * batch;
		if ((pos / mpAligned) % 2 == 0) memcpy(&Ahost[mpos], &n1, sizeof(Ahost[0]));
		else memcpy(&Ahost[mpos], &n2, sizeof(Ahost[0]));
	    }
	}

	for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                int pos = i * k + j;
		int mpos = i * k + j + n * k * batch;
		if ((pos / mpAligned) % 2 == 0) memcpy(&Bhost[mpos], &n2, sizeof(Bhost[0]));
		else memcpy(&Bhost[mpos], &n1, sizeof(Bhost[0]));
	    }
	}
    }
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__half, __half, float>::randomData() {
    std::mt19937 engine(std::random_device{}());
    std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(dist(engine));
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(dist(engine));
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(dist(engine));
}


template <>
inline void TestBench<__half, __half, cuComplex>::randomData() {
    std::mt19937 engine(std::random_device{}());
    std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(dist(engine)/100.);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(dist(engine)/100.);
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(dist(engine) + 1);
}
