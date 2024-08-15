#include <stdio.h>
#include <cuda_runtime.h>


__global__ void indices(){
	printf("Thread(%d,%d,%d) Block: (%d,%d,%d) Dim: (%d,%d,%d)\n",
		threadIdx.x,threadIdx.y,threadIdx.z, 
		blockIdx.x,blockIdx.y,blockIdx.z,
		blockDim.x,blockDim.y,blockDim.z);
}

int main(int argc, char const *argv[])
{
  dim3 grid(2,2);
  dim3 block(2,2);
	// Inicializamos los arreglos requeridos
  indices<<<grid,block>>>();
	// esperamos a que los hilos terminen
  cudaDeviceSynchronize();
	 return 0;
}
