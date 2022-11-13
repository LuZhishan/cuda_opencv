#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define N 5

__global__ void gpuSquare(float *d_in, float *d_out) 
{
	int tid = threadIdx.x;
    d_out[tid] = d_in[tid]*d_in[tid];
}

int main()
{
    float h_in[N], h_out[N];
    for (size_t i = 0; i < N; i++)
    {
        h_in[i] = i;
    }
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    gpuSquare<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < N; i++)
    {
        std::cout << h_out[i] << std::endl;
    }
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}