#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define Nx 4
#define Ny 4


void initialData(float *h_A, int size){
	// Definimos la matriz
	for (int i = 0; i < size; ++i){
		h_A[i] = i + 1;
	}

}

void printMatrix(float *h_A, int rows, int cols){
	float *Mat = h_A;
	printf("Imprime matriz: \n\n");
	for (int j = 0; j < rows; ++j){
		for (int i = 0; i < cols; ++i){
			printf("%.1f ",Mat[i]);
		}
		Mat += cols;
		printf("\n");
	}
}


__global__ void matrixOnGPU(float *d_A,int nx){
	// Accedemos a memoria de cada elemento de la matriz
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	int idy = threadIdx.y + (blockDim.y * blockIdx.y);
	// Obtenemos el indice donde se localiza el elemento a ser procesado
	int ix = idy * nx + idx;
	// Cada indice accede a sus matriz y a sus elementos
	printf("PHilo(%d,%d): PBlock (%d,%d) (idx,idy) (%d,%d)  elemento %d valor %f \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,idx,idy,ix,d_A[ix]);
}



int main(int argc, char const *argv[])
{
  A[10][10];
  // Numero de elementos totales
	int nElemns    = Nx*Ny;
	size_t nBytes  = nElemns * sizeof(float); 
	// Definimos una matriz lineal en C
	float *h_A;
	h_A = (float *)malloc(nBytes);
	// Inicializamos la matriz
	initialData(h_A, nElemns);
	printMatrix(h_A,Nx,Ny);
	// Paso 1. Solicitamos memoria al Device
	float *d_A;
	cudaMalloc((float**)&d_A, nBytes);
	// Paso 2. Enviamos la informacion del host al device
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	// Paso 3. Lanzamos el kernel
	dim3  bloques(2,2);
	dim3 hilos(2,2);
	matrixOnGPU<<<bloques,hilos>>>(d_A,Ny);
	// Liberamos memoria 
	free(h_A);
	cudaFree(d_A);
	return 0;
}
