#include <stdio.h>
#include <sys/time.h>

#define Nx 5000
#define Ny 5000

int MA[Nx][Ny];
int MB[Nx][Ny];

// Función para medir tiempo
double cpuTime(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}



void inicializa_row_major_order(int A[Nx][Ny],int rows, int cols){
    int cuenta = 0;
    for (int i = 0; i < rows; ++i){
        for(int j=0; j < cols; ++j){
            A[i][j] = cuenta;
            cuenta++;
        }
    }

}


void inicializa_colum_major_order(int A[Nx][Ny],int rows, int cols){
    int cuenta = 0;
    for (int i = 0; i < rows; ++i){
        for(int j=0; j < cols; ++j){
            A[j][i] = cuenta;
            cuenta++;
        }
    }

}




void showMatrix(int A[Nx][Ny],int rows, int cols){
    for (int i = 0; i < rows; ++i){
        for(int j=0; j < cols; j++ ){
            printf("%d ",A[i][j]);
        }
        printf("\n");
    }
}


int main(int argc, char const *argv[]){
    // Defimos dos matrices dinamicas

    double tic,toc,tictoc;
    tic = toc = tictoc = 0.0;
    // Inicializacion de la mnatrix
    tic = cpuTime();
    inicializa_row_major_order(MA,Nx,Ny);
    toc = cpuTime();
    tictoc = toc - tic;
    printf("Elapsed time row major order: %lf\n",tictoc);
    // Muestra matriz en pantalla
    //showMatrix(MB,Nx,Ny);
    // Inicializa matrix 
    tic = toc = tictoc = 0.0;
    tic = cpuTime();
    inicializa_colum_major_order(MB,Nx,Ny);
    toc = cpuTime();
    tictoc = toc - tic;
    printf("Elapsed time col major order: %lf\n",tictoc);
    // Muestra matriz en pantalla
    //showMatrix(MA,Nx,Ny);
    return 0;
}