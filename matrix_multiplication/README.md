# GPU-Based Matrix Multiplication

## Overview
This program performs matrix multiplication on the GPU using various optimization strategies. It reads two input matrices **A** and **B** from binary files, computes their product **C**, and writes the result to output files. The program benchmarks seven different implementations using CUDA and cuBLAS, and reports execution time and numerical errors.

## File Structure
- `matrix_multiplication.cu` : Main GPU matix multiplication benchmarking program
- `generate_random_matrix.cu` : Generates random input matrices and saves them as binary files
- `measure_host_time.h` : Utility for host-side timing
- `Makefile` : Build script for both matrix multiplication and matrix generator binaries

## Input/Output of Matrix Multiplication
Matrices are stored in binary format:
- First 4 bytes: number of rows (int)
- Next 4 bytes: number of columns (int)
- Remaining: `rows × cols` values in row-major order as:
    - `float` for FP32 kernels
    - `__half` for Tensor Core kernels

## Features
This project supports matrix multiplication accelerated by the GPU. It includes seven CUDA-based matrix multiplication methods.
In descriptions, “FP16 → FP32” means the kernel takes FP16 inputs but produces FP32 outputs.
1. Global memory only (no shared memory) (FP32)
2. Shared memory optimization (FP32)
3. Shared memory + More-Work-Per-Thread (MWPT) (FP32)
4. Tensor Core without shared memory (FP16 -> FP32)
5. Tensor Core with shared memory (FP16 -> FP32)
6. cuBLAS with CUDA Cores (FP32)
7. cuBLAS with Tensor Cores (FP16 -> FP32)

## Performance Evaluation
Each kernel is timed over multiple runs (default: 20 repetitions).
Output matrix **C** from each method is compared with a CPU-based reference using:
- Average relative error
- Maximum relative error

## Execution
### 1. Compile the programs
```bash
make
```
### 2. Generate input matrices
```bash
./matgen 1024 1024 2048     # Format: ./randmat m n k
```
![image](https://github.com/user-attachments/assets/221f8b47-7515-4f20-a201-95ed669d182e)

### 3. Run multiplication and benchmark
```bash
./matmul
```
![image](https://github.com/user-attachments/assets/0083acea-5334-4cfa-bedf-ad6e7d67d14b)
![image](https://github.com/user-attachments/assets/006e4905-1307-4932-be34-2e5af05bffd5)

