//#include "kernel.cuh"
#include <stdio.h>
#include <stdlib.h> 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int GenerateRand(int rmax)
{
	return ((int)rand() / (int)(RAND_MAX)) * rmax;
}

void GenerateVector(int *v, int vecSz)
{
	for (int i = 0; i < vecSz; i++)
		v[i] = GenerateRand(100);
}

__global__
void AddInt_Kernel(int *a_d, int *b_d, int *c_d, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n)
		c_d[i] = a_d[i] + b_d[i];
}

cudaError_t CudaSetup(const int *a, const int *b, int *c, int vecSz)
{
	int *a_d = 0;
	int *b_d = 0;
	int *c_d = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		printf("ERROR: cudaSetDevice failed!\n");

	cudaStatus = cudaMalloc((void**)&c_d, vecSz * sizeof(int));
	cudaStatus = cudaMalloc((void**)&a_d, vecSz * sizeof(int));
	cudaStatus = cudaMalloc((void**)&b_d, vecSz * sizeof(int));

	cudaStatus = cudaMemcpy(a_d, a, vecSz * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(b_d, b, vecSz * sizeof(int), cudaMemcpyHostToDevice);

	if (cudaStatus != cudaSuccess)
		printf("ERROR");

	//Kernel launch
	dim3 DimGrid(vecSz / 256, 1, 1);
	if (vecSz % 256) DimGrid.x++;
	dim3 DimBlock(256, 1, 1);
	AddInt_Kernel<<<DimGrid, DimBlock>>>(a_d, b_d, c_d, vecSz); //Unsure about DimGrid and Dimblock
	//AddInt_Kernel << <ceil(vecSz / 256), 256 >> >(a_d, b_d, c_d, vecSz);

	//AddInt_Kernel<<<1, vecSz >>>(a_d, b_d, c_d, vecSz);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	cudaStatus = cudaMemcpy(c, c_d, vecSz * sizeof(int), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	return cudaStatus;
}

int main()
{
	srand((unsigned int)time(NULL));

	const int vecSz = 5;
	/*int *a = (int*)malloc(vecSz * sizeof(int));
	int *b = (int*)malloc(vecSz * sizeof(int));
	int *c = NULL;*/

	const int a[vecSz] = { 1, 2, 3, 4, 5 };
	const int b[vecSz] = { 10, 20, 30, 40, 50 };
	int c[vecSz] = { 0 };

	/*GenerateVector(a, vecSz);
	GenerateVector(b, vecSz);*/

	CudaSetup(a, b, c, vecSz);

	printf("C result:");
	for (int i = 0; i < vecSz; i++)
		printf(" %d", c[i]);

	printf("\nDone..\n");

	cudaDeviceReset();
	getchar();

	return 0;
}