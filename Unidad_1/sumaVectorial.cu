#include <stdio.h>
#include <stdlib.h>

#define N 90000000

void initialData(float *A, int size){
  srand(0);
  for(int i=0; i<size;i++){
    A[i] = rand() % 100 + 1;
  }
}


void sumaOnCPU(float *A, float *B, float *C, int size){
  for(int i=0; i<size; i++){
    C[i] = A[i] + B[i];
  }
}

void showVector(float *A, int size){
  for(int i=0; i<size; i++){
    printf("V[%d] = %f\n",i,A[i]);
  }
}


__global__ void sumaOnGPU(float *dA, float *dB, float *dC, int size){
  int idx = threadIdx.x;
  dC[idx] = dA[idx]+dB[idx];
}





int main(){
  // Definimos vector
  size_t nBytes = N * sizeof(float);
  float *h_A, *h_B, *h_C,*h_res;
  // asignar memopria
  h_A   = (float *)malloc(nBytes);
  h_B   = (float *)malloc(nBytes);
  h_C   = (float *)malloc(nBytes);
  h_res = (float *)malloc(nBytes);
  // Paso1. Asignamos memoria en el GPU
  float *d_A, *d_B, *d_C;
  cudaMalloc((float**)&d_A,nBytes);
  cudaMalloc((float**)&d_B,nBytes);
  cudaMalloc((float**)&d_C,nBytes);
  // Inicializa vector
  initialData(h_A,N);
  initialData(h_B,N);
  // Inicizamos con cero
  memset(h_C,0,nBytes);
  memset(h_res,0,nBytes);
  // paso 2. Enviar datos
  cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_C,h_C,nBytes,cudaMemcpyHostToDevice);
  // Paso 3. Realizar la suma 
  sumaOnGPU<<<1,N>>>(d_A,d_B,d_C,N);
  // Pass 4. 
  cudaMemcpy(h_res,d_C,nBytes,cudaMemcpyDeviceToHost);
  // Sumamos
  sumaOnCPU(h_A,h_B,h_C,N);
  // Imprime resultado
  showVector(h_res,N);
  printf("\n\n");
  showVector(h_C,N);
  // Liberar memoria
  free(h_A);
  free(h_B);
  free(h_C);
  // Paso 5. Liberar memoria
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}