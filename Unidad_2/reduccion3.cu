#include <stdio.h>

#define N 10000

void initialData(int *h_A, int size){
	for (int i = 0; i < size; ++i){
		h_A[i] = i + 1;
	}
}

__global__ void reduceInterleaved (int *input, int *output, int size) {
  // Obtenermos el identificador de Hilo
  int tid = threadIdx.x;
  // Calculamos el elemento que tendriamos que procesar del vectos
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Cada bloque debe apuntar al inicio de su fragmento de codigo
  int *idata = input + blockIdx.x * blockDim.x;
  // Los hilos sobrantes deben ser inhabilitados
  if(idx >= size){
    return;
  }
  // Obtenemos la suma por pares entrelazados
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        idata[tid] += idata[tid + stride];
    }
    // Sincronizamos la ejecucion de los hilos
    __syncthreads();
  }
  // Almacenamos el valor calculado
  if (tid == 0)[
    output[blockIdx.x] = idata[0];
  ]
}


int main(int argc, char const *argv[]){
	// Obtenemos las propiedades del GPU
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Identificador del GPU = %d, GPU name = %s \n", dev, deviceProp.name);
	// Configuracion del kernel
	dim3 hilos(512);
	dim3 bloques((N+hilos.x-1)/hilos.x);
	printf("Configuracion -> Bloques: (%d) Hilos: (%d) \n",bloques.x,hilos.x);
	//  Asignamos memoria en el CPU
	size_t nBytesArray   = N * sizeof(int);
  size_t nBytesParcial = bloques.x * sizeof(int);
	int *h_Input, *h_Output;
	h_Input  = (int *) malloc(nBytesArray); // Definimos el arreglo total
  h_Output = (int *) malloc(nBytesParcial);
  initialData(h_Input, N);
  memset(h_Output,0,nBytesParcial);
  // Paso 1. Solicitamos espacio en el GPU
  int *d_Input, *d_Output;
  cudaMalloc((void **)&d_Input, nBytesArray);
  cudaMalloc((void **)&d_Output, nBytesParcial);
  // Paso 2. Copiamos los datos del CPU al GPU
  cudaMemcpy(d_Input, h_Input, nBytesArray, cudaMemcpyHostToDevice);
  // Paso 3. Lanzamos el kernel
  reduceInterleaved<<<bloques,hilos>>>(d_Input,d_Output,N);
  cudaDeviceSynchronize();
  // Paso 4. Recibimos la informacion del GPU
  cudaMemcpy(h_Output, d_Output, nBytesParcial, cudaMemcpyDeviceToHost);
  // Realizamos una suma secuencial
  int sumaTotal = 0;
  for(int i=0; i<bloques.x; ++i){
      sumaTotal += h_Output[i];
  }
  printf("Suma total: %d \n",sumaTotal);
  // Paso 5.  Liberamos la memoria solciictada
  free(h_Input);
  free(h_Output);
  cudaFree(d_Input);
  cudaFree(d_Output);
	return 0;
}