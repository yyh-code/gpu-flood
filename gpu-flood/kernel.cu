#include <stdio.h>
#include <malloc.h>
#include <windows.h>
#include <math.h>
#include <stdlib.h>
#include "../common/book.h"
#include "cuda.h"
#include "../common/cpu_bitmap.h"
#include <iostream>
#include <time.h>
int M;
int N;
double dx, dy;
double xllcorner;
double yllcorner;
int nodata;
int ny, nx;
char ncols[15];
char nrows[15];
char xllcorner_label[15];
char yllcorner_label[15];
char cellsize[15];
char NODATA_value[15];
int *z;
int *infa;
int *fa_pre;

void gridread()    /*读取流向文件*/
{
	int i, j;
	FILE *fp;
	char infile[10];
	printf("输入流向文件名：");
	scanf("%s", infile);
	fp = fopen(infile, "r");
	if (fp == NULL)
	{
		printf("cannot open file\n");
		return;
	}
	fscanf(fp, "%s %d", &ncols, &M);
	fscanf(fp, "%s %d", &nrows, &N);
	fscanf(fp, "%s %lf", &xllcorner_label, &xllcorner);
	fscanf(fp, "%s %lf", &yllcorner_label, &yllcorner);
	fscanf(fp, "%s %lf", &cellsize, &dx);
	fscanf(fp, "%s %d", &NODATA_value, &nodata);
	dy = dx;
	z = (int*)calloc(N*M, sizeof(int));
	infa = (int*)calloc(N*M, sizeof(int));
	fa_pre = (int*)calloc(N*M, sizeof(int));
	for (i = 0; i<N; i++)
	{
		for (j = 0; j<M; j++)
		{
			fscanf(fp, "%d ", &z[i*M + j]);
			infa[i*M + j] = 1;
			fa_pre[i*M + j] = 0;
		}
		fscanf(fp, "\n");
	}
	fclose(fp);
}
__global__ void flood_cal(int *fd, int *fa, int *fa_pre, int rows, int cols, int *count, int i, int *dev_countx) {
	int icol = blockIdx.x*blockDim.x + threadIdx.x;
	int irow = blockIdx.y*blockDim.y + threadIdx.y;
	if (irow >= rows || icol >= cols)
		return;
	int self = irow * cols + icol;
	int nie, nise, nis, nisw, niw, ninw, nin, nine;
	int count_self = 0, fa_tra1 = 0, fa_pre1 = 0;
	nie = self + 1;
	nise = self + cols + 1;
	nis = self + cols;
	nisw = self + cols - 1;
	niw = self - 1;
	ninw = self - cols - 1;
	nin = self - cols;
	nine = self - cols + 1;
	count_self = count[self];
	__syncthreads();
	if (count_self == i) {
		fa_tra1 = fa[self];
		fa_pre1 = fa_pre[self];
		if (icol <cols - 1 && fd[self] == 1) {
			atomicAdd(&fa[nie], fa_tra1 - fa_pre1);
			count[nie] = count_self + 1;
		}
		if (irow<rows - 1 && icol<cols - 1 && fd[self] == 2)
		{
			atomicAdd(&fa[nise], fa_tra1 - fa_pre1);
			count[nise] = count_self + 1;
		}
		if (irow<rows - 1 && fd[self] == 4) {
			atomicAdd(&fa[nis], fa_tra1 - fa_pre1);
			count[nis] = count_self + 1;
		}
		if (irow<rows - 1 && icol>0 && fd[self] == 8) {
			atomicAdd(&fa[nisw], fa_tra1 - fa_pre1);
			count[nisw] = count_self + 1;
		}
		if (icol>0 && fd[self] == 16) {
			atomicAdd(&fa[niw], fa_tra1 - fa_pre1);
			count[niw] = count_self + 1;
		}
		if (irow >0 && icol >0 && fd[self] == 32) {
			atomicAdd(&fa[ninw], fa_tra1 - fa_pre1);
			count[ninw] = count_self + 1;
		}
		if (irow>0 && fd[self] == 64) {
			atomicAdd(&fa[nin], fa_tra1 - fa_pre1);
			count[nin] = count_self + 1;
		}
		if (irow >0 && icol <cols - 1 && fd[self] == 128) {
			atomicAdd(&fa[nine], fa_tra1 - fa_pre1);
			count[nine] = count_self + 1;
		}
		__syncthreads();
		fa_pre[self] = fa_tra1;
		atomicAdd(dev_countx, 1);
	}
	fa_pre1 = 0, fa_tra1 = 0, count_self = 0;
}



int main() {

	clock_t start_time = clock();
	gridread();
	printf("In process\n");
	cudaEvent_t  start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));
	int fulsize = N*M;
	int *dev_fa, *outfa, *dev_z, *dev_fa_pre;
	int BLOCKCOLS = 16;
	int BLOCKROWS = 16;
	int gridCols = (M + BLOCKCOLS - 1) / BLOCKCOLS;
	int gridRows = (N + BLOCKROWS - 1) / BLOCKROWS;
	dim3 dimgrid(gridCols, gridRows);
	dim3 dimblock(BLOCKCOLS, BLOCKROWS);
	int *count, *count_pre, *count1;
	outfa = (int*)malloc(fulsize * sizeof(int));
	count1 = (int*)malloc(fulsize * sizeof(int));
	int count_x, *dev_countx;
	HANDLE_ERROR(cudaMalloc((void**)&dev_fa, fulsize * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_fa_pre, fulsize * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_z, fulsize * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&count, fulsize * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_countx, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(dev_fa, infa, fulsize * sizeof(int), cudaMemcpyHostToDevice));//传输初始值到设备数组
	HANDLE_ERROR(cudaMemcpy(dev_fa_pre, fa_pre, fulsize * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_z, z, fulsize * sizeof(int), cudaMemcpyHostToDevice));//传输流向数据到设备
	for (int k = 0;; k++)
	{
		count_x = 0;
		cudaMemcpy(dev_countx, &count_x, sizeof(int), cudaMemcpyHostToDevice);
		flood_cal << <dimgrid, dimblock >> >(dev_z, dev_fa, dev_fa_pre, N, M, count, k, dev_countx);
		cudaMemcpy(&count_x, dev_countx, sizeof(int), cudaMemcpyDeviceToHost);
		if (count_x == 0)
		{
			printf("迭代次数：%d\n", k);
			break;
		}
	}
	HANDLE_ERROR(cudaMemcpy(outfa, dev_fa, fulsize * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(count1, count, N*M * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time to generate:    %3.2f ms\n", elapsedTime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	//输出文件
	FILE *outfile;
	outfile = fopen("out_flood.txt", "w");
	if (outfile == NULL)
	{
		printf("cannot open the file\n");
	}

	fprintf(outfile, "%s       %d\n", "ncols", M);
	fprintf(outfile, "%s       %d\n", "nrows", N);
	fprintf(outfile, "%s 	%.12lf\n", "xllcorner", xllcorner);
	fprintf(outfile, "%s 	%.12lf\n", "yllcorner", yllcorner);
	fprintf(outfile, "%s 	%.12lf\n", "cellsize", dx);
	fprintf(outfile, "%s  	%d\n", "NODATA_value", nodata);
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<M; j++)
		{
			fprintf(outfile, "%d ", outfa[i*M + j]);
		}
		fprintf(outfile, "\n");
	}

	fclose(outfile);
	printf("finished!\n");
	clock_t end_time = clock();
	float clockTime = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000;
	//printf("Running time is:   %3.2f ms\n", clockTime);
	cudaFree(infa);
	cudaFree(fa_pre);
	cudaFree(dev_z);
	cudaFree(count);
	free(outfa);
	free(count1);
	free(z);
	free(infa);
}