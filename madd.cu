/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <stdio.h>

__global__ void kernel_madd(float* A, float* B, float* C, int M, int N);
__global__ void kernel_msub(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mcoef(float* A, float* C, int M, int N, float coef);
__global__ void kernel_mtile(float* A, float* C, int M, int N);
__global__ void kernel_mabs(float* A, float* C, int M, int N);
__global__ void kernel_msum(float* A, float* C, int M, int N);
__global__ void kernel_msum2(float* A, float* C, int M, int N);
__global__ void kernel_mmul(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mdiv(float* A, float* B, float* C, int M, int N);
__global__ void kernel_mtrans(float* A, float* C, int M, int N);
__global__ void kernel_maddCons(float* A, float* C, int M, int N, float constant);
__global__ void kernel_mmatmul(float* A, float* B, float* C, int M, int N);

void cu_madd(float* A, float* B, float* C, int M, int N)
{
	float* d_a, * d_b, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(float) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

	kernel_madd << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_madd(float* A, float* B, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;

	if (ix < M && iy < N)
		C[idx] = A[idx] + B[idx];
}



void cu_msub(float* A, float* B, float* C, int M, int N)
{
	float* d_a, * d_b, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(float) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

	kernel_msub << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_msub(float* A, float* B, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;

	if (ix < M && iy < N)
		C[idx] = A[idx] - B[idx];
}



void cu_mcoef(float* A, float* C, int M, int N, float coef)
{
	float* d_a, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;


	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_c, sizeof(float) * M * N);

	cudaMemcpy(d_a, A, sizeof(float) * N, cudaMemcpyHostToDevice);

	kernel_mcoef << < grid, blk >> > (d_a, d_c, M, N, coef);

	cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
}

__global__ void kernel_mcoef(float* A, float* C, int M, int N, float coef) {

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;
	printf("%d\n", idx);

	if (idx < M*N)
		C[idx] = coef * A[idx];
}



void cu_mtile(float* A, float* C, int M, int N)
{
	float* d_a, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;


	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_c, sizeof(float) * M * N);

	cudaMemcpy(d_a, A, sizeof(float) * N, cudaMemcpyHostToDevice);

	kernel_mtile << < grid, blk >> > (d_a, d_c, M, N);

	cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
}

__global__ void kernel_mtile(float* A, float* C, int M, int N) {

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;


	if (ix < M && iy < N)
		C[idx] = A[idx % N];
}



void cu_mabs(float* A, float* C, int M, int N)
{
	float* d_a, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(float) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);

	kernel_mabs << < grid, blk >> > (d_a, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
}

__global__ void kernel_mabs(float* A, float* C, int N, int M) {

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;


	if (ix < M && iy < N) {
		if (A[idx] >= 0)
			C[idx] = A[idx];

		if (A[idx] < 0)
			C[idx] = -1 * A[idx];
	}

}



void cu_msum(float* A, float* C, int M, int N)
{
	float* d_a, * d_b, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 1;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = 1;
	grid.z = 1;

	int size = sizeof(float) * N * M;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_c, size / N);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	kernel_msum << < grid, blk >> > (d_a, d_c, M, N);


	cudaMemcpy(C, d_c, size / N, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
}

__global__ void kernel_msum(float* A, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;


	float sum = 0;
	if (ix < M) {
		for (int i = 0; i < N; i++) {
			if (ix + i * M < M * N)
				sum = sum + A[ix + i * M];
		}
		C[ix] = sum;
	}
}



void cu_msum2(float* A, float* C, int M, int N)
{
	float* d_a, * d_b, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 1;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = 1;
	grid.z = 1;

	int size = sizeof(float) * N * M;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_c, size / M);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	kernel_msum2 << < grid, blk >> > (d_a, d_c, M, N);


	cudaMemcpy(C, d_c, size / M, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
}

__global__ void kernel_msum2(float* A, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;


	float sum = 0;
	if (ix < N) {
		for (int i = 0; i < M; i++) {
			if (ix *M+ i < M*N)
				sum = sum + A[ix*M + i];
		}
		C[ix] = sum;
	}
}



void cu_mmul(float* A, float* B, float* C, int M, int N)
{
	float* d_a, * d_b, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(float) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

	kernel_mmul << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_mmul(float* A, float* B, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;


	if (ix < M && iy < N)
		C[idx] = A[idx] * B[idx];
}



void cu_mdiv(float* A, float* B, float* C, int M, int N)
{
	float* d_a, * d_b, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(float) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

	kernel_mdiv << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_mdiv(float* A, float* B, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;


	if (ix < M && iy < N)
		C[idx] = (A[idx] / B[idx]);
}



void cu_mtrans(float* A, float* C, int M, int N)
{
	float* d_a, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(float) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);

	kernel_mtrans << < grid, blk >> > (d_a, d_c, M, N);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
}

__global__ void kernel_mtrans(float* A, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;


	if (ix < M && iy < N)
		C[ix * N + iy] = A[iy * M + ix];
}



void cu_maddCons(float* A, float* C, int M, int N, float constant)
{
	float* d_a, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(float) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);

	kernel_maddCons << < grid, blk >> > (d_a, d_c, M, N, constant);

	cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);
}

__global__ void kernel_maddCons(float* A, float* C, int M, int N, float constant)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;


	if (ix < M && iy < N)
		C[idx] = A[idx] + constant;
}



void cu_mmatmul(float* A, float* B, float* C, int M, int N, int dim)
{
	float* d_a, * d_b, * d_c;

	dim3 blk;
	blk.x = 16; blk.y = 16;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;


	cudaMalloc((void**)&d_a, sizeof(float) * N * dim);
	cudaMalloc((void**)&d_b, sizeof(float) * dim * M);
	cudaMalloc((void**)&d_c, sizeof(float) * N*M);

	cudaMemcpy(d_a, A, sizeof(float) * N*dim , cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, sizeof(float) * dim * M, cudaMemcpyHostToDevice);

	kernel_mmatmul << < grid, blk >> > (d_a, d_b, d_c, M, N);

	cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_mmatmul(float* A, float* B, float* C, int M, int N)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * M + ix;


	if (idx < M*N)
		C[idx] = A[(int)idx/M] * B[idx%M];
}