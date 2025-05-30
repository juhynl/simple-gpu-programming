#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas.h>

#include "measure_host_time.h"

#define FILE_A "my_file_A.bin"
#define FILE_B "my_file_B.bin"
#define FILE_A_HF "my_file_A_hf.bin"
#define FILE_B_HF "my_file_B_hf.bin"
#define FILE_C_1 "my_file_C_1.bin"
#define FILE_C_2 "my_file_C_2.bin"
#define FILE_C_3 "my_file_C_3.bin"
#define FILE_C_4 "my_file_C_4.bin"
#define FILE_C_5 "my_file_C_5.bin"
#define FILE_C_6 "my_file_C_6.bin"
#define FILE_C_7 "my_file_C_7.bin"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define NUM_REPEATS 100

using namespace nvcuda;

// File read/write functions
void fread_matrix(const char *filename, float **matrix, int *row, int *col);
void fread_matrix(const char *filename, __half **matrix, int *row, int *col);
void fwrite_matrix(const char *filename, float *matrix, int row, int col);

// Matrix multiplication kernels
__global__ void mm_naive_cc(const float *A, const float *B, float *C, const int M, const int N, const int K);

template <const uint TS>
__global__ void mm_sm_cc(const float *A, const float *B, float *C, const int M, const int N, const int K);

template <const uint TS, const uint WPT, const uint RTS>
__global__ void mm_sm_mwpt_cc(const float *A, const float *B, float *C, const int M, const int N, const int K);

template <const uint WMMA_M, const uint WMMA_N, const uint WMMA_K>
__global__ void mm_naive_tc(__half *A, __half *B, float *C, const int M, const int N, const int K);

template <const uint TILE_M, const uint TILE_N, const uint TILE_K, const uint WMMA_M, const uint WMMA_N, const uint WMMA_K>
__global__ void mm_sm_tc(__half *A, __half *B, float *C, const int M, const int N, const int K);

// Host functions
void MM_DEVICE_GM(const float *A, const float *B, float *C, const int M, const int N, const int K);
void MM_DEVICE_SM(const float *A, const float *B, float *C, const int M, const int N, const int K);
void MM_DEVICE_SM_MWPT(const float *A, const float *B, float *C, const int M, const int N, const int K);
void MM_DEVICE_TC_GM(__half *A, __half *B, float *C, const int M, const int N, const int K);
void MM_DEVICE_TC_SM(__half *A, __half *B, float *C, const int M, const int N, const int K);
void MM_DEVICE_CUBLAS_CC(const float *A, const float *B, float *C, const int M, const int N, const int K);
void MM_DEVICE_CUBLAS_TC(const __half *A, const __half *B, float *C, const int M, const int N, const int K);
void MM_HOST_DOUBLE(const float *A, const float *B, float *C, const int M, const int N, const int K);
double compute_avg_relative_err(const float *A, const float *B, int M, int N);
double find_max_relative_err(const float *A, const float *B, int M, int N);

// Function pointer type for matrix multiplication device functions
typedef void (*MM_DEVICE_FUNC)(const void *, const void *, float *, int, int, int);

// Handle for cuBLAS
cublasHandle_t handle_global;

int main(void)
{
    float *A, *B, *C;
    __half *A_hf, *B_hf;
    int m, n, k;

    // Read input matrices from binary files
    fread_matrix(FILE_A, &A, &m, &k);
    fread_matrix(FILE_B, &B, &k, &n);
    fread_matrix(FILE_A_HF, &A_hf, &m, &k);
    fread_matrix(FILE_B_HF, &B_hf, &k, &n);
    C = (float *)malloc(m * n * sizeof(float));

    printf("\nM: %d, N: %d, K: %d\n\n", m, n, k);

    // Copy FP32 matrices to device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * n * sizeof(float));

    // Copy __half matrices to device memory
    __half *d_A_hf, *d_B_hf;

    cudaMalloc(&d_A_hf, m * k * sizeof(__half));
    cudaMalloc(&d_B_hf, k * n * sizeof(__half));

    cudaMemcpy(d_A_hf, A_hf, m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_hf, B_hf, k * n * sizeof(__half), cudaMemcpyHostToDevice);

    // Function pointers to the corresponding matrix multiplication implementations
    MM_DEVICE_FUNC funcs[] = {
        (MM_DEVICE_FUNC)MM_DEVICE_GM,
        (MM_DEVICE_FUNC)MM_DEVICE_SM,
        (MM_DEVICE_FUNC)MM_DEVICE_SM_MWPT,
        (MM_DEVICE_FUNC)MM_DEVICE_TC_GM,
        (MM_DEVICE_FUNC)MM_DEVICE_TC_SM,
        (MM_DEVICE_FUNC)MM_DEVICE_CUBLAS_CC,
        (MM_DEVICE_FUNC)MM_DEVICE_CUBLAS_TC,
    };

    // Label each functions
    const char *func_types[] = {
        "CUDA Cores/float",
        "CUDA Cores/float/shared memory",
        "CUDA Cores/float/shared memory/MWPT",
        "Tensor Cores/half",
        "Tensor Cores/half/shared memory",
        "CUDA Cores/float/cuBLAS",
        "Tensor Cores/half/cuBLAS",
    };

    // Output file names for saving result matrices corresponding to each method
    const char *output_files[] = {
        FILE_C_1, FILE_C_2, FILE_C_3, FILE_C_4, FILE_C_5, FILE_C_6, FILE_C_7};

    // Device pointers to input matrices A and B
    const void *d_A_list[] = {d_A, d_A, d_A, d_A_hf, d_A_hf, d_A, d_A_hf};
    const void *d_B_list[] = {d_B, d_B, d_B, d_B_hf, d_B_hf, d_B, d_B_hf};

    cublasCreate_v2(&handle_global);

    double total_time_ms[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // Test seven matrix multiplication functions
    for (int i = 0; i < 7; i++)
    {
        // Initialize output matrix C to zero on device memory
        cudaMemset(d_C, 0, m * n * sizeof(float));

        // Warm-up
        for (int iter = 0; iter < 10; iter++)
        {
            funcs[i](d_A_list[i], d_B_list[i], d_C, m, n, k);
        }

        // Repeat NUM_REPEATS times to measure average execution time
        for (int iter = 0; iter < NUM_REPEATS; iter++)
        {
            CHECK_TIME_START(_start);
            funcs[i](d_A_list[i], d_B_list[i], d_C, m, n, k);
            CHECK_TIME_END(_start, _end, _compute_time);
            total_time_ms[i] += _compute_time;
        }

        // Copy result matrix from device to host and save it to a file
        cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        fwrite_matrix(output_files[i], C, m, n);
    }

    // Run host-side matrix multiplication using double precision for reference
    MM_HOST_DOUBLE(A, B, C, m, n, k);

    for (int i = 0; i < 7; i++)
    {
        float *C_result;
        int m_result, n_result;

        // Compute average execution time over NUM_REPEATS
        double avg_time = total_time_ms[i] / NUM_REPEATS;

        // Read result matrix generated by the i-th GPU method from file
        fread_matrix(output_files[i], &C_result, &m_result, &n_result);

        // Compute average and maximum absolute relative errors between GPU and CPU results
        double relative_err = compute_avg_relative_err(C, C_result, m_result, n_result);
        double max_err = find_max_relative_err(C, C_result, m_result, n_result);

        // Print results
        printf("[%d] GPU time(%s) = %e(ms)\n\n", i + 1, func_types[i], avg_time);
        printf("\t[Absolute relative errors] average = %e, max = %e\n\n", relative_err, max_err);

        free(C_result);
    }

    // Destory cuBLAS handle
    cublasDestroy_v2(handle_global);

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_hf);
    cudaFree(d_B_hf);

    free(A);
    free(B);
    free(A_hf);
    free(B_hf);

    return 0;
}

// ================================================================================================================
// File read/write functions
// ================================================================================================================
void fread_matrix(const char *filename, float **matrix, int *row, int *col)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        printf("fread_matrix Failed!\n");
        exit(1);
    }
    else
    {
        fread(row, sizeof(int), 1, fp);
        fread(col, sizeof(int), 1, fp);
        *matrix = (float *)malloc(*row * *col * sizeof(float));
        fread(*matrix, sizeof(float), *row * *col, fp);
    }
    fclose(fp);
}

void fread_matrix(const char *filename, __half **matrix, int *row, int *col)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        printf("fread_matrix Failed! %s\n", filename);
        exit(1);
    }
    else
    {
        fread(row, sizeof(int), 1, fp);
        fread(col, sizeof(int), 1, fp);
        *matrix = (__half *)malloc(*row * *col * sizeof(__half));
        fread(*matrix, sizeof(__half), *row * *col, fp);
    }
    fclose(fp);
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

// ================================================================================================================
// Matrix multiplication kernels
// ================================================================================================================
__global__ void mm_naive_cc(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t_m = tid / N;
    int t_n = tid % N;

    if (t_m >= M || t_n >= N)
        return;

    float accum = 0.0f;
    for (int k = 0; k < K; k++)
    {
        accum += A[t_m * K + k] * B[k * N + t_n];
    }
    C[t_m * N + t_n] = accum;
}

template <const uint TS>
__global__ void mm_sm_cc(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    __shared__ float tile_A[TS * TS];
    __shared__ float tile_B[TS * TS];

    int tile_row = blockIdx.x / CEIL_DIV(N, TS);
    int tile_col = blockIdx.x % CEIL_DIV(N, TS);

    int tid = threadIdx.x;
    int local_row = threadIdx.x / TS;
    int local_col = threadIdx.x % TS;

    int global_row = tile_row * TS + local_row;
    int global_col = tile_col * TS + local_col;

    if (global_row >= M || global_col >= N)
        return;

    float accum = 0.0f;
    int A_row_idx = global_row;
    int B_col_idx = global_col;
    for (int k = 0; k < K; k += TS)
    {
        int A_col_idx = k + local_col;
        int B_row_idx = k + local_row;

        tile_A[tid] = (A_col_idx < K) ? A[A_row_idx * K + A_col_idx] : 0.0f;
        tile_B[local_col * TS + local_row] = (B_row_idx < K) ? B[B_row_idx * N + B_col_idx] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TS; i++)
        {
            accum += tile_A[local_row * TS + i] * tile_B[local_col * TS + i];
        }
        __syncthreads();
    }
    C[global_row * N + global_col] = accum;
}

template <const uint TS, const uint WPT, const uint RTS>
__global__ void mm_sm_mwpt_cc(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    __shared__ float tile_A[TS * TS];
    __shared__ float tile_B[TS * TS];
    float accum[WPT];
    for (int i = 0; i < WPT; i++)
    {
        accum[i] = 0.0f;
    }

    int tile_row = blockIdx.x / CEIL_DIV(N, TS);
    int tile_col = blockIdx.x % CEIL_DIV(N, TS);

    int tid = threadIdx.x;
    int local_row = threadIdx.x / TS;
    int local_col = threadIdx.x % TS;

    int global_row = tile_row * TS + local_row;
    int global_col = tile_col * TS + local_col;

    if (global_row >= M || global_col >= N)
        return;

    for (int k = 0; k < K; k += TS)
    {
        int A_col_idx = k + local_col;
        int B_row_idx = k + local_col;

        for (int w = 0; w < WPT; w++)
        {
            int A_row_idx = global_row + w * RTS;
            int B_col_idx = tile_col * TS + w * RTS + local_row;

            tile_A[tid + w * RTS * TS] = (A_col_idx < K) ? A[A_row_idx * K + A_col_idx] : 0.0f;
            tile_B[tid + w * RTS * TS] = (B_row_idx < K) ? B[B_row_idx * N + B_col_idx] : 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TS; i++)
        {
            float tmp = tile_B[local_col * TS + i];
            for (int w = 0; w < WPT; w++)
            {
                accum[w] += tile_A[w * RTS * TS + local_row * TS + i] * tmp;
            }
        }
    }
    for (int w = 0; w < WPT; w++)
    {
        C[global_row * N + global_col + w * RTS * N] = accum[w];
    }
}

template <const uint WMMA_M, const uint WMMA_N, const uint WMMA_K>
__global__ void mm_naive_tc(__half *A, __half *B, float *C, const int M, const int N, const int K)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int wid = tid / warpSize;
    int warp_row = wid / CEIL_DIV(N, WMMA_N);
    int warp_col = wid % CEIL_DIV(N, WMMA_N);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K)
    {
        __half *A_start = A + warp_row * WMMA_M * K + k;
        __half *B_start = B + k * N + warp_col * WMMA_N;
        wmma::load_matrix_sync(a_frag, A_start, K);
        wmma::load_matrix_sync(b_frag, B_start, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    float *C_start = C + warp_row * WMMA_M * N + warp_col * WMMA_N;
    wmma::store_matrix_sync(C_start, c_frag, N, wmma::mem_row_major);
}

template <const uint TILE_M, const uint TILE_N, const uint TILE_K, const uint WMMA_M, const uint WMMA_N, const uint WMMA_K>
__global__ void mm_sm_tc(__half *A, __half *B, float *C, const int M, const int N, const int K)
{

    __shared__ __align__(128) __half shared_memory[TILE_M * TILE_K + TILE_K * TILE_N];

    __half *A_shared = shared_memory;
    __half *B_shared = shared_memory + TILE_M * TILE_K;

    int tile_row = blockIdx.x / CEIL_DIV(N, TILE_N);
    int tile_col = blockIdx.x % CEIL_DIV(N, TILE_N);

    int wid = threadIdx.x / warpSize;
    int wrow = wid / CEIL_DIV(TILE_N, WMMA_N);
    int wcol = wid % CEIL_DIV(TILE_N, WMMA_N);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += TILE_K)
    {
        for (int i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x)
        {
            A_shared[i] = A[(tile_row * TILE_M + (i / TILE_K)) * K + (i % TILE_K) + k];
        }
        for (int i = threadIdx.x; i < TILE_K * TILE_N; i += blockDim.x)
        {
            B_shared[i] = B[(k + (i / TILE_N)) * N + tile_col * TILE_N + (i % TILE_N)];
        }
        __syncthreads();

        for (int i = 0; i < TILE_K; i += WMMA_K)
        {
            wmma::load_matrix_sync(a_frag, &A_shared[wrow * WMMA_M * TILE_K + i], TILE_K);
            wmma::load_matrix_sync(b_frag, &B_shared[i * TILE_N + wcol * WMMA_N], TILE_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
    }
    wmma::store_matrix_sync(&C[(tile_row * TILE_M + wrow * WMMA_M) * N + tile_col * TILE_N + wcol * WMMA_N], c_frag, N, wmma::mem_row_major);
}

// ================================================================================================================
// Host Functions
// ================================================================================================================
void MM_DEVICE_GM(const float *A, const float *B, float *C, const int M, const int N, const int K)
{

    int size_C = M * N;
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(size_C, blockDim.x));
    mm_naive_cc<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void MM_DEVICE_SM(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    const uint TS = 32;
    dim3 blockDim(TS * TS);
    dim3 gridDim(CEIL_DIV(M, TS) * CEIL_DIV(N, TS));
    mm_sm_cc<TS><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void MM_DEVICE_SM_MWPT(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    const uint TS = 16;
    const uint WPT = 8;
    const uint RTS = TS / WPT;
    dim3 blockDim(TS * RTS);
    dim3 gridDim(CEIL_DIV(M, TS) * CEIL_DIV(N, TS));
    mm_sm_mwpt_cc<TS, WPT, RTS><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void MM_DEVICE_TC_GM(__half *A, __half *B, float *C, const int M, const int N, const int K)
{
    const uint WMMA_M = 16;
    const uint WMMA_N = 16;
    const uint WMMA_K = 16;
    dim3 blockDim(256);
    int warp_per_block = blockDim.x / 32;
    dim3 gridDim(CEIL_DIV(CEIL_DIV(M, WMMA_M) * CEIL_DIV(N, WMMA_N), warp_per_block));
    mm_naive_tc<WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void MM_DEVICE_TC_SM(__half *A, __half *B, float *C, const int M, const int N, const int K)
{
    const uint TILE_M = 64;
    const uint TILE_N = 32;
    const uint TILE_K = 16;
    const uint WMMA_M = 16;
    const uint WMMA_N = 16;
    const uint WMMA_K = 16;
    const uint shm_size = TILE_M * TILE_K + TILE_K * TILE_N;
    dim3 blockDim((TILE_M / WMMA_M * TILE_N / WMMA_N) * 32);
    dim3 gridDim(CEIL_DIV(M, TILE_M) * CEIL_DIV(N, TILE_N));

    mm_sm_tc<TILE_M, TILE_N, TILE_K, WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim, shm_size>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void MM_DEVICE_CUBLAS_CC(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    float alpha = 1.0, beta = 0.0;

    cublasGemmEx(handle_global, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void MM_DEVICE_CUBLAS_TC(const __half *A, const __half *B, float *C, const int M, const int N, const int K)
{
    float alpha = 1.0, beta = 0.0;
    cublasGemmEx(handle_global, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void MM_HOST_DOUBLE(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            double accum = 0.0;
            for (int k = 0; k < K; k++)
            {
                accum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = (float)accum;
        }
    }
}

double compute_avg_relative_err(const float *A, const float *B, int M, int N)
{
    double error = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        double diff = fabs(A[i] - B[i] / A[i]);
        error += diff;
    }

    return error / (M * N);
}

double find_max_relative_err(const float *A, const float *B, int M, int N)
{
    double max_err = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        double diff = fabs(A[i] - B[i] / A[i]);
        if (diff > max_err)
        {
            max_err = diff;
        }
    }

    return max_err;
}