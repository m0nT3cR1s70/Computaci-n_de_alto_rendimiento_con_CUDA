#include <stdio.h>


void hiOnCPU(){
  printf("Hola mundo desde CPU!!!!!\n");
}

__global__ void hiOnGPU(){
  printf("Hola mundo desde GPU\n!!!!!");
}

int main(){
  hiOnCPU();
  // Llamamos al kernel
  hiOnGPU<<<3,3>>>();
  cudaDeviceSynchronize();
  // Devuelve valor
  return 0;
}