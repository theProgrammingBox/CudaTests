#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int main() {
    int N = 8;
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // Initialize a and b
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // Copy a and b to the device
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Run the kernel
    add<<<1, N>>>(dev_a, dev_b, dev_c);

    // Copy c back to the host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free the memory on the device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
