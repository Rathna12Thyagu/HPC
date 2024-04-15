#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int conjugategradient_p(double *A, double *b, double *x, int n, int max_iterations, int num_threads);
int conjugategradient_s(double *A, double *b, double *x, int n, int max_iterations);

int main(int argc, char *argv[])
{
    int max_iterations = 10; // Fixed to 10

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
        conjugategradient_s(matrix, b, x, dimension, max_iterations);
        serial_time = omp_get_wtime() - serial_time;

        double parallel_time = omp_get_wtime();
        conjugategradient_p(matrix, b, x, dimension, max_iterations, num_threads);
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

int conjugategradient_p(double *A, double *b, double *x, int n, int max_iterations, int num_threads)
{
    // Parallel implementation of Conjugate Gradient method
    double r[n];
    double p[n];
    double px[n];

#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }

    double alpha = 0;
    while (max_iterations--)
    {
        double sum = 0;
#pragma omp parallel for reduction(+ : sum) num_threads(num_threads)
        for (int i = 0; i < n; i++)
        {
            sum += r[i] * r[i];
        }

        double temp[n];
#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++)
        {
            temp[i] = 0;
        }

        double num = 0;
#pragma omp parallel for reduction(+ : num) num_threads(num_threads)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i] += A[i * n + j] * p[j];
            }
            num += temp[i] * p[i];
        }

        alpha = sum / num;

#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++)
        {
            px[i] = x[i];
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * temp[i];
        }

        double beta = 0;
#pragma omp parallel for reduction(+ : beta) num_threads(num_threads)
        for (int i = 0; i < n; i++)
        {
            beta += r[i] * r[i];
        }
        beta = beta / sum;

#pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }

        int c = 0;
        for (int i = 0; i < n; i++)
        {
            if (r[i] < 0.000001)
                c++;
        }
        if (c == n)
            break;
    }

    return 0;
}

int conjugategradient_s(double *A, double *b, double *x, int n, int max_iterations)
{
    // Serial implementation of Conjugate Gradient method
    double r[n];
    double p[n];
    double px[n];

    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }

    double alpha = 0;
    while (max_iterations--)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += r[i] * r[i];
        }

        double temp[n];
        for (int i = 0; i < n; i++)
        {
            temp[i] = 0;
        }

        double num = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i] += A[i * n + j] * p[j];
            }
            num += temp[i] * p[i];
        }

        alpha = sum / num;

        for (int i = 0; i < n; i++)
        {
            px[i] = x[i];
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * temp[i];
        }

        double beta = 0;
        for (int i = 0; i < n; i++)
        {
            beta += r[i] * r[i];
        }
        beta = beta / sum;

        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }

        int c = 0;
        for (int i = 0; i < n; i++)
        {
            if (r[i] < 0.000001)
                c++;
        }
        if (c == n)
            break;
    }

    return 0;
}
