#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

int backsubstitution_p(double *A, double *b, double *x, int n, int max_iterations);
int backsubstitution_s(double *A, double *b, double *x, int n, int max_iterations);

int main(int argc, char *argv[])
{
    int num_threads;
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    omp_set_num_threads(num_threads);

    int max_iterations = 10;

    int start_dimension = 5000;
    int end_dimension = 10000;
    int interval_dimension = (end_dimension - start_dimension) / 20;

    for (int dimension = start_dimension; dimension <= end_dimension; dimension += interval_dimension)
    {
        printf("\n\nMatrix Dimension: %d x %d\n", dimension, dimension);

        double *matrix = malloc(sizeof(double) * dimension * dimension);
        double *b = malloc(sizeof(double) * dimension);
        double *x = malloc(sizeof(double) * dimension);

        for (int i = 0; i < (dimension * dimension); i++)
        {
            double x = (rand() / (double)RAND_MAX);
            matrix[i] = x;
        }

        for (int i = 0; i < dimension; i++)
        {
            double x = (rand() / (double)RAND_MAX);
            b[i] = x;
        }

        double *matrix_copy = malloc(sizeof(double) * dimension * dimension);
        double *b_copy = malloc(sizeof(double) * dimension);
        double *x_copy = malloc(sizeof(double) * dimension);

        memcpy(matrix_copy, matrix, sizeof(double) * dimension * dimension);
        memcpy(b_copy, b, sizeof(double) * dimension);
        memcpy(x_copy, x, sizeof(double) * dimension);

        double serial_time = omp_get_wtime();
        backsubstitution_p(matrix, b, x, dimension, max_iterations);
        serial_time = omp_get_wtime() - serial_time;

        double parallel_time = omp_get_wtime();
        backsubstitution_s(matrix_copy, b_copy, x_copy, dimension, max_iterations);
        parallel_time = omp_get_wtime() - parallel_time;

        printf("Serial Time: %f seconds\n", serial_time);
        printf("Parallel Time: %f seconds\n", parallel_time);

        free(matrix);
        free(b);
        free(x);
        free(matrix_copy);
        free(b_copy);
        free(x_copy);
    }

    return 0;
}

int backsubstitution_p(double *A, double *b, double *x, int n, int max_iterations)
{
    int i, j;
    double temp;
    for (i = n - 1; i >= 0; i--)
    {
        temp = b[i];
#pragma omp parallel for reduction(- : temp) shared(A, b, x) private(j)
        for (j = i + 1; j < n; j++)
        {
            temp -= A[i * n + j] * x[j];
        }
        x[i] = temp / A[i * n + i];
    }

    return 0;
}

int backsubstitution_s(double *A, double *b, double *x, int n, int max_iterations)
{
    int i, j;
    double temp;
    for (i = n - 1; i >= 0; i--)
    {
        temp = b[i];
        for (j = i + 1; j < n; j++)
        {
            temp -= A[i * n + j] * x[j];
        }
        x[i] = temp / A[i * n + i];
    }

    return 0;
}