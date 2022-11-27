#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 50000

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("thread id is %d, block id is %d, tid is %d\n", threadIdx.x, blockIdx.x, tid);
    while (tid < N)
    {
        d_c[tid] = d_a[tid] + d_b[tid]; // 因为是多线程同时添加，所以这里相当于一次性添加了一个grid(512*512)的数据
        tid += blockDim.x * gridDim.x;  // 因为现在只有一个维度，所以后续就要在下一个网格分配数据了
    }
}

int main()
{
	int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

	for (int i = 0; i < N; i++)
    {
		h_a[i] = 2 * i * i;
		h_b[i] = i;
	}

	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));
	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 这里描述的是一个grid，包含512个block，每个block里有512个线程然后根据上面开辟的内存分配grid的数量
    gpuAdd<<<512, 512>>>(d_a, d_b, d_c); 
	cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); // 等待所有线程执行完成

    int Correct = 1;
	for (int i = 0; i < N; i++) 
    {
		if ((h_a[i] + h_b[i] != h_c[i]))
		{
			Correct = 0;
            break;
		}
	}
    printf("GPU has computed Sum Correctly\n");

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}