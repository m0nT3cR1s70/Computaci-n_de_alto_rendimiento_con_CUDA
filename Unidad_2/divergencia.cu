#include <stdio.h>
#include <stdlib.h> 
#include <sys/time.h>

#define N 64


double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}



__global__ void kernel_1(float *c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (idx % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;

    c[idx] = a + b;
}

__global__ void kernel_2(float *c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    int warpSize = 32;
    if ((idx / warpSize) % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;


    c[idx] = a + b;
}

__global__ void kernel_3(float *c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    bool ipred = (idx % 2 == 0);

    if (ipred)
        a = 100.0f;


    if (!ipred)
        b = 200.0f;


    c[idx] = a + b;
}


__global__ void warmingup(float *c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    int warpSize = 32;
    if ((idx / warpSize) % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;

    c[idx] = a + b;
}





int main(int argc, char const *argv[]){
	// Variables para medir tiempo
	double tic, toc, tictoc = 0;
	// Calculamos el numero de Bytes para almacenar un vector
	size_t nBytes = N * sizeof(float);
	// Configuracion del kernel
	dim3 hilos(1024);
	dim3 bloques((N + hilos.x-1)/hilos.x);
	// Paso 1. Solicitamos espacio en el GPU
	float *d_A;
	cudaMalloc((float**)&d_A,nBytes);
	// Paso 3. Invocamos los kernels
	tic = cpuTime();
	warmingup<<<bloques,hilos>>>(d_A);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictoc = toc - tic;
	printf("Elapsed time for warmingup: %lf\n",tictoc);
  // Kernel 1
	tic = cpuTime();
	kernel_1<<<bloques,hilos>>>(d_A);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictoc = toc - tic;
	printf("Elapsed time for Kernel_1: %lf\n",tictoc); 
	// Kernel 2
	tic = cpuTime();
	kernel_1<<<bloques,hilos>>>(d_A);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictoc = toc - tic;
	printf("Elapsed time for Kernel_2: %lf\n",tictoc);
	// Kernel 3
	tic = cpuTime();
	kernel_3<<<bloques,hilos>>>(d_A);
	cudaDeviceSynchronize();
	toc = cpuTime();
	tictoc = toc - tic;
	printf("Elapsed time for Kernel_3: %lf\n",tictoc);
	// Paso 5. Liberamos la memoria solicitada en el GPU
	cudaFree(d_A);
	return 0;
}