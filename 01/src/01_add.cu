#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) 
{
	*d_c = *d_a + *d_b;
}

int main()
{
	int h_a = 1, h_b = 2, h_c;  	// 定义变量用于接收GPU的数据
    int *d_a, *d_b, *d_c;	// 定义一个指针
	// 在GPU上开辟内存
	cudaMalloc((void**)&d_a, sizeof(int)); 
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMalloc((void**)&d_c, sizeof(int));
	// 将数据拷贝至GPU
	cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
	// 调用GPU函数
	gpuAdd <<<1, 1>>> (d_a, d_b, d_c);	
	// 从GPU取回数据
	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost); 
	// 释放GPU资源
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	std::cout << h_c << std::endl;
}