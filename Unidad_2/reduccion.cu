#include <stdio.h>

#define N 6

void initialData(int *h_A, int size){
	for (int i = 0; i < size; ++i){
		h_A[i] = i + 1;
	}
}


__global__ void reduceNeighbored(int *input, int *output, int size){
  // Identificador de hilo/
  int tid  = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Cada hilo sabra donde inicia el pedazo de memoria que le corresponde acceder
  int *idata = input + blockIdx.x * blockDim.x;
  // Definimos las fronteras
  if(idx >= size){
    return;
  }
  // Realizamos las primeras sumas parciales
  for(int stride = 1; stride < blockDim.x; stride *=2){
    if ((tid % (2*stride)) == 0){
      idata[tid] += idata[tid + stride];
    }
    // BARRERA DE SINCRONIZACION
    __syncthreads();
  }
  //
  if(tid == 0){ 
    output[blockIdx.x] = idata[0];
  } 
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
  reduceNeighbored<<<bloques,hilos>>>(d_Input,d_Output,N);
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