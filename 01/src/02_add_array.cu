#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define N 10

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) 
{
	int tid = blockIdx.x;
	if (tid < N)
		d_c[tid] = d_a[tid] + d_b[tid];
}

int main()
{
    int h_a[N], h_b[N], h_c[N]; 
    for (size_t i = 0; i < N; i++)
    {
        h_a[i] = 2*i*i;
        h_b[i] = i ;
	}
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));
	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    // 这里的N是N个块block, 1是每个块内一个线程
	gpuAdd <<<N, 1>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i++)
    {
        std::cout << h_c[i] << std::endl;
    }
    
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}