#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        exit(-1);
    }
}

__global__ void DevRand(
    uint32_t seed1, uint32_t seed2,
    float* tensor, uint32_t widthA, uint32_t heightA,
    uint32_t stride1D, uint32_t stride2D, uint32_t batches) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < widthA * heightA * batches) {
        int32_t Hash = idx;
        Hash ^= seed1;
        Hash *= 0xBAC57D37;
        Hash ^= seed2;
        Hash *= 0x24F66AC9;

        uint32_t x = idx % widthA;
        uint32_t y = (idx / widthA) % heightA;
        uint32_t z = idx / (widthA * heightA);
        tensor[x + y * stride1D + z * stride2D] = Hash * 0.0000000004656612875245796f;
    }
}

struct GpuRand {
    uint32_t seed1, seed2;

    GpuRand() {
        seed1 = time(NULL) ^ 0xE621B963;
        seed2 = 0x6053653F ^ (time(NULL) >> 32);

        printf("Seed1: %u\n", seed1);
        printf("Seed2: %u\n\n", seed2);
    }

    void Rand(
        float* tensor, uint32_t widthA, uint32_t heightA,
        uint32_t stride1D, uint32_t stride2D, uint32_t batches) {
        seed1 ^= seed2;
        seed1 *= 0xBAC57D37;
        seed2 ^= seed1;
        seed2 *= 0x24F66AC9;
        DevRand<<<ceil(0.0009765625f * widthA * heightA * batches), 1024>>> (
            seed1, seed2,
            tensor, widthA, heightA,
            stride1D, stride2D, batches);
    }
};

void PrintDevTensor(
    float* tensor, uint32_t widthA, uint32_t heightA,
    uint32_t stride1D, uint32_t stride2D, uint32_t batches,
    const char* label, bool transposed)
    {
    float* hostArr = (float*)malloc(widthA * heightA * batches * sizeof(float));
    cudaMemcpy(hostArr, tensor, widthA * heightA * batches * sizeof(float), cudaMemcpyDeviceToHost);

    uint32_t b, h, w;
    uint32_t* hh, * ww;

    hh = &h;
    ww = &w;
    if (transposed) {
        hh = &w;
        ww = &h;
        uint32_t temp = heightA;
        heightA = widthA;
        widthA = temp;
    }
    printf("%s:\n", label);
    for (b = 0; b < batches; b++) {
        for (h = 0; h < heightA; h++) {
            for (w = 0; w < widthA; w++) {
                printf("%6.3f ", hostArr[*ww + *hh * stride1D + b * stride2D]);
            }
            printf("\n");
        }
        printf("\n");
    }
    free(hostArr);
}

__global__ void DevGemm0(
    uint32_t widthD, uint32_t heightA, uint32_t widthA,
    float* tensorA, uint32_t stride1DA, uint32_t stride2DA,
    float* tensorB, uint32_t stride1DB, uint32_t stride2DB,
    float* tensorC, uint32_t stride1DC, uint32_t stride2DC,
    float* tensorD, uint32_t stride1DD, uint32_t stride2DD,
    uint32_t batches)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < widthD * heightA * batches) {
        uint32_t x = idx % widthD;
        uint32_t y = idx / widthD;

        for (uint32_t b = 0; b < batches; b++) {
            float sum = tensorC[x + y * stride1DC + b * stride2DC];
            for (uint32_t i = 0; i < widthA; i++) {
                sum += tensorA[i + y * stride1DA + b * stride2DA] * tensorB[x + i * stride1DB + b * stride2DB];
            }
            tensorD[x + y * stride1DD + b * stride2DD] = sum;
        }
    }
}

__global__ void DevGemm1(
    uint32_t widthD, uint32_t heightA, uint32_t widthA,
    float* tensorA, uint32_t stride1DA, uint32_t stride2DA,
    float* tensorB, uint32_t stride1DB, uint32_t stride2DB,
    float* tensorC, uint32_t stride1DC, uint32_t stride2DC,
    float* tensorD, uint32_t stride1DD, uint32_t stride2DD,
    uint32_t batches)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < widthD * heightA * batches) {
        uint32_t x = idx % widthD;
        uint32_t y = idx / widthD;

        float* ptrA = tensorA + y * stride1DA;
        float* ptrB = tensorB + x;
        float* ptrC = tensorC + x + y * stride1DC;
        float* ptrD = tensorD + x + y * stride1DD;

        for (uint32_t b = 0; b < batches; b++) {
            float sum = *ptrC;
            for (uint32_t i = 0; i < widthA; i++) {
                sum += *ptrA * *ptrB;
                ptrA++;
                ptrB += stride1DB;
            }
            *ptrD = sum;

            ptrA += stride2DA - widthA;
            ptrB += stride2DB - widthA * stride1DB;
            ptrC += stride2DC;
            ptrD += stride2DD;
        }
    }
}

int main()
{
    const uint32_t widthA = 4;
    const uint32_t heightA = 3;
    const uint32_t widthD = 2;
    const uint32_t batches = 2;

    GpuRand rand;

    float *devTensorA, *devTensorB, *devTensorC, *devTensorD;
    checkCudaStatus(cudaMalloc(&devTensorA, widthA * heightA * batches * sizeof(float)));
    checkCudaStatus(cudaMalloc(&devTensorB, widthD * widthA * batches * sizeof(float)));
    checkCudaStatus(cudaMalloc(&devTensorC, widthD * heightA * batches * sizeof(float)));
    checkCudaStatus(cudaMalloc(&devTensorD, widthD * heightA * batches * sizeof(float)));

    rand.Rand(devTensorA, widthA, heightA, widthA, widthA * heightA, batches);
    rand.Rand(devTensorB, widthD, widthA, widthD, widthD * widthA, batches);
    rand.Rand(devTensorC, widthD, heightA, widthD, widthD * heightA, batches);

    PrintDevTensor(devTensorA, widthA, heightA, widthA, widthA * heightA, batches, "A", false);
    PrintDevTensor(devTensorB, widthD, widthA, widthD, widthD * widthA, batches, "B", false);
    PrintDevTensor(devTensorC, widthD, heightA, widthD, widthD * heightA, batches, "C", false);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 100; i--;)
    DevGemm1<<<1, widthD * heightA>>>(
        widthD, heightA, widthA,
        devTensorA, widthA, widthA * heightA,
        devTensorB, widthD, widthD * widthA,
        devTensorC, widthD, widthD * heightA,
        devTensorD, widthD, widthD * heightA,
        batches);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= 10.0f;

    PrintDevTensor(devTensorD, widthD, heightA, widthD, widthD * heightA, batches, "D", false);

    printf("Elapsed time: %f ms\n\n", milliseconds);

    // calculate gflops
    float flops = 2.0f * widthA * widthD * heightA * batches;
    printf("GFLOPS: %f\n", flops / milliseconds / 1000.0f);

    return 0;
}