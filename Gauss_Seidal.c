#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int seidel_p(double A[], double b[], double x[], int n, int max_iterations, int num_threads);
int seidel_s(double A[], double b[], double x[], int n, int max_iterations);

int main(int argc, char *argv[])
{
    int max_iterations = 5;

    int start_dimension = 5000;
    int end_dimension = 10000;
    int interval_dimension = (end_dimension - start_dimension) / 20;

    int num_threads;
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    omp_set_num_threads(num_threads);

    for (int dimension = start_dimension; dimension <= end_dimension; dimension += interval_dimension)
    {
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

        double serial_time = omp_get_wtime();
        seidel_s(matrix, b, x, dimension, max_iterations);
        serial_time = omp_get_wtime() - serial_time;

        double parallel_time = omp_get_wtime();
        seidel_p(matrix, b, x, dimension, max_iterations, num_threads);
        parallel_time = omp_get_wtime() - parallel_time;

        printf("Matrix Dimension: %d x %d\n", dimension, dimension);
        printf("Serial Time: %f seconds\n", serial_time);
        printf("Parallel Time: %f seconds\n", parallel_time);

        free(matrix);
        free(b);
        free(x);
    }

    return 0;
}

int seidel_p(double A[], double b[], double x[], int n, int max_iterations, int num_threads)
{
    int i, j, k;
    double dxi;
    double epsilon = 1.0e-4;

    for (k = 0; k < max_iterations; k++)
    {
#pragma omp parallel for num_threads(num_threads) private(i, j, dxi) shared(A, b, x) schedule(static)
        for (i = 0; i < n; i++)
        {
            dxi = b[i];
            for (j = 0; j < n; j++)
            {
                if (j != i)
                {
                    dxi -= A[i * n + j] * x[j];
                }
            }
            x[i] = dxi / A[i * n + i];
        }
    }

    return 0;
}

int seidel_s(double A[], double b[], double x[], int n, int max_iterations)
{
    int i, j, k;
    double dxi;
    double epsilon = 1.0e-4;

    for (k = 0; k < max_iterations; k++)
    {
        for (i = 0; i < n; i++)
        {
            dxi = b[i];
            for (j = 0; j < n; j++)
            {
                if (j != i)
                {
                    dxi -= A[i * n + j] * x[j];
                }
            }
            x[i] = dxi / A[i * n + i];
        }
    }

    return 0;
}
