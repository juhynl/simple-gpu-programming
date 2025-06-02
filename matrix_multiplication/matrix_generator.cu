#include <stdio.h>
#include "cuda_runtime.h"
#include "cublas.h"
#include "string.h"
#include <sys/time.h>

#include "measure_host_time.h"

#define FILE_A "my_file_A.bin"
#define FILE_B "my_file_B.bin"
#define FILE_A_HF "my_file_A_hf.bin"
#define FILE_B_HF "my_file_B_hf.bin"

void randomize_matrix_s(int N, float *M)
{
    struct timeval time{};

    gettimeofday(&time, nullptr);
    srand(time.tv_usec);

    for (int i = 0; i < N; i++)
    {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        M[i] = tmp;
    }
}

void fwrite_matrix(const char *filename, float *matrix, int row, int col)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        printf("fwrite_matrix Failed!\n");
        exit(1);
    }
    else
    {
        fwrite(&row, sizeof(int), 1, fp);
        fwrite(&col, sizeof(int), 1, fp);
        fwrite(matrix, sizeof(float), row * col, fp);
    }
    fclose(fp);
}

void fwrite_matrix(const char *filename, __half *matrix, int row, int col)
{
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        printf("fwrite_matrix Failed!\n");
        exit(1);
    }
    else
    {
        fwrite(&row, sizeof(int), 1, fp);
        fwrite(&col, sizeof(int), 1, fp);
        fwrite(matrix, sizeof(__half), row * col, fp);
    }
    fclose(fp);
}

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        printf("Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));

    randomize_matrix_s(m * k, A);
    randomize_matrix_s(k * n, B);

    fwrite_matrix(FILE_A, A, m, k);
    fwrite_matrix(FILE_B, B, k, n);

    __half *A_hf = (__half *)malloc(m * k * sizeof(__half));
    __half *B_hf = (__half *)malloc(k * n * sizeof(__half));

    for (int i = 0; i < m * k; i++)
    {
        A_hf[i] = __float2half(A[i]);
    }

    for (int i = 0; i < k * n; i++)
    {
        B_hf[i] = __float2half(B[i]);
    }

    fwrite_matrix(FILE_A_HF, A_hf, m, k);
    fwrite_matrix(FILE_B_HF, B_hf, k, n);

    free(A);
    free(B);
    free(A_hf);
    free(B_hf);

    return 0;
}