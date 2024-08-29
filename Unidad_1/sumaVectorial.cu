#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 90000000


double cpuTime(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-06);
}


void compareResults(float *res_GPU, float *res_CPU, int size){
  double epsilon = 1.0E-8;
  int flag = 1;
  for(int i=0; i<size; i++){
    if(abs(res_GPU[i]-res_CPU[i])>epsilon){
      printf("ERROR: suma distinta\n");
      flag = 0;
      break;
    }
  }
  if(flag == 1){
    printf("SUMA CORRECTA\n");
  }
}





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
  //int idx = threadIdx.x;
  //int idx = blockIdx.x;
  // Patron acceso vectorial
  int idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if(idx < size){
      dC[idx] = dA[idx]+dB[idx];
  }
}




int main(){
  // Variables para medir tiempo 
  double tic,toc, tictocS = 0, tictocP = 0;
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
  dim3 hilos(1024); 
  dim3 bloque((N+hilos.x-1)/hilos.x);    // 87890.625 bloques -> (87891)(1024) = 90,000,384 
  tic = cpuTime();
  sumaOnGPU<<<bloque,hilos>>>(d_A,d_B,d_C,N);
  
  toc = cpuTime();
  tictocP = toc - tic;
  printf("Elapsed time for GPU: %lf segundos\n",tictocP);
  // Pass 4. 
  cudaMemcpy(h_res,d_C,nBytes,cudaMemcpyDeviceToHost);
  // Sumamos
  tic = cpuTime();
  sumaOnCPU(h_A,h_B,h_C,N);
  toc = cpuTime();
  tictocS = toc - tic;
  printf("Elapsed time for CPU: %lf segundos\n",tictocS);
  printf("Speed-up: %lf \n",tictocS/tictocP);
  // Imprime resultado
  //showVector(h_res,N);
  compareResults(h_res,h_C,N);
  //printf("\n\n");
  //showVector(h_C,N);
  // Liberar memoria
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_res);
  // Paso 5. Liberar memoria
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}