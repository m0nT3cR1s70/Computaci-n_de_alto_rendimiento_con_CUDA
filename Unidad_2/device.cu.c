#include <stdio.h>
#include <cuda_runtime.h>


int main(int argc, char *argv[]){
    int device = 0;
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    printf("Dispositivo %d: %s\n", device, properties.name);
    printf("SMs: %d\n", properties.multiProcessorCount);
    printf("Memoria constante: %4.2f KB\n", properties.totalConstMem / 1024.0);
    printf("Memoria compartida por bloque: %4.2f KB\n",properties.sharedMemPerBlock / 1024.0);
    printf("Tamanio del warp:  %d\n",properties.warpSize);
    printf("Hilos por bloque maximo: %d\n",properties.maxThreadsPerBlock);
    printf("Hilos maximos por SM:  %d\n", properties.maxThreadsPerMultiProcessor);
    printf("Numero de warps por SM:    %d\n",properties.maxThreadsPerMultiProcessor / 32);
    return EXIT_SUCCESS;
}
