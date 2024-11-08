#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Definimos el numero de puntos utilizado
#define N 10

// Error cuadratico medio
float rms(float *exacta, float *aprox, int size){
	float rmss = 0.0;
	for (int i = 0; i < size; ++i){
	 	rmss += (aprox[i] - exacta[i]) * (aprox[i] - exacta[i]);
	 } 
	 rmss = sqrt(rmss/size);
	 return rmss;
}






// Función para resolver un sistema de ecuaciones lineales usando eliminación de Gauss
void Gauss(int size, float A[size][size], float b[size], float x[size]) {
    // Eliminación hacia adelante
    for (int i = 0; i < size; i++) {
        // Seleccionar el pivote y evitar división entre 0
        if (fabs(A[i][i]) < 1e-10) {
            printf("Error: El pivote es cero. El sistema no puede resolverse.\n");
            return;
        }

        for (int k = i + 1; k < size; k++) {
            float factor = A[k][i] / A[i][i];
            for (int j = i; j < size; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Sustitución hacia atrás
    for (int i = size - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < size; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}


// Ponemos una funcion para almacernar el resultado en un archivo de texto
void saveData(const char *filename, float *coordX, float *sol, int size) {
    // Abrir el archivo en modo escritura
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("No se pudo abrir el archivo %s\n", filename);
        return;
    }

    // Escribir los valores en el archivo
    fprintf(file, "X Y\n");
    for (int i = 0; i < size; i++) {
        fprintf(file, "%lf %lf\n", coordX[i], sol[i]); // Escribe una línea con valores de ambos arreglos
    }

    // Cerrar el archivo
    fclose(file);
}


// Imprimimos el contenido de un vector
void showVector(float *V, int size){
	printf("\n");
	for (int i = 0; i < size; ++i){
		printf("%lf ",V[i]);
	}
	printf("\n");
}


void showMatrix(int size, float A[size][size]){
	printf("\n");
	for (int i = 0; i < size; ++i){
		for (int j = 0; j < size; ++j){
			printf("%lf  ",A[i][j]);
		}
		printf("\n");
	}
}


// Discretizamos el dominio
void linespace(float x0, float x1, float *dominio, int nPuntos){
	// Calculamos el delta de x denominado h en este caso
	float h = (x1 - x0) / (nPuntos - 1);
	// Obtenemos los puntos restantes
	for (int i = 0; i < nPuntos; ++i){
		dominio[i] = h*i;
	}
}

// Llenamos la matriz
void laplaciano1D( int size, float A[size][size]){
	// Inicializamos la matriz
	for (int i = 0; i < size; ++i){
		for (int j = 0; j < size; ++j){
			A[i][j] = 0.0;
		}
	}
	// Llenamos la matriz tridiagonal
	// Primer renglon
	A[0][0] = -2.0;
	A[0][1] =  1.0;
	for(int i = 1; i < size-1; ++i){
		A[i][i-1] =  1.0;
		A[i][i]   = -2.0;
		A[i][i+1] =  1.0;
	}
	// Renglon final
	A[size-1][size-2] =  1.0;
	A[size-1][size-1] = -2.0;
}

// Vector b
void V_b(float *b,float u0, float u1, int size){
	// Llenamos el primer elemento del vector
	b[0] = -u0;
	// Cuerpo del vector
	for (int i = 1; i < size-1; ++i){
		b[i] = 0.0;
	}
	b[size-1] = -u1;
}
// Calculamos la solucion exacta
void analitica(float *sol, float *coordX, int size){
	for (int i = 0; i < size; ++i){
		sol[i] = - coordX[i] + 2;
	}
}

int main(int argc, char const *argv[]){
	// Numeros de puntos utilizados
	float coordX[N];
	float exacta[N];
	// Definimos las propiedades del problema
	// Coordenadas espaciales
	float x0 = 0.0;
	float x1 = 1.0;
	// Condiciones de frontera
	float u0 = 2.0;
	float u1 = 1.0;
	// PASO 1. DISCRETIZAMOS EL DOMINIO
	linespace(x0,x1,coordX,N);
	// Calculamos la solucion exacta
	analitica(exacta,coordX,N);
	saveData("C:/Users/marni/Documents/exacta.txt",coordX,exacta,N);
	//showVector(coordX,N);
	// PASO 2. GENERAMOS EL SISTEMA DE ECUACIONS Ax = b
	float A[N][N];
	float b[N];
	float x[N];
	laplaciano1D(N,A);
	V_b(b,u0,u1,N);
	//showMatrix(N,A);
	//showVector(b,N);
	// PASO 3. OBTENEMOS LA SOLUCION DEL SISTEMA DE ECUACIONES
	Gauss(N,A,b,x);
	saveData("C:/Users/marni/Documents/aproximada.txt",coordX,x,N);
	//showVector(exacta,N);
	//showVector(x,N);
	printf("RMS = %lf \n",rms(exacta,x,N));
	return 0;
}