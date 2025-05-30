#include <stdio.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "measure_host_time.h"

#define RAND_UNIFORM ((float)rand() / RAND_MAX)
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ===============================================================================================
// Experiment configurations
// ===============================================================================================
#define NUM_ITER_FOR_TEST 20
#define N (1 << 24) // Number of points
#define BLOCK_SIZE 256
// ===============================================================================================

// ===============================================================================================
// CUDA kernels
// ===============================================================================================
__global__ void init_curand_states(curandState *states, int n, unsigned long seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n)
        return;

    // each thread gets a different state
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void do_MonteCarlo_simulation(curandState *states, float *counts, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if the thread ID is within bounds
    if (tid >= n)
        return;

    // Initialize the count to 0
    counts[tid] = 0;

    // Get the state for this thread
    curandState localState = states[tid];

    // Generate a random float between -1.0 and 1.0
    float x = 2.0f * curand_uniform(&localState) - 1.0f;
    float y = 2.0f * curand_uniform(&localState) - 1.0f;
    float z = 2.0f * curand_uniform(&localState) - 1.0f;

    // Check if the generated point is inside the unit sphere
    if (x * x + y * y + z * z <= 1.0f)
        counts[tid] = 1;

    // Save the state back: unnecessary unless this kernel is called again
    states[tid] = localState;
}

__global__ void reduce1(float *x, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if the thread ID is within bounds
    if (tid >= n)
        return;

    // Perform a summation
    float tsum = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int k = tid; k < n; k += stride)
    {
        tsum += x[k];
    }

    // Store the partial sum in the corresponding position of x
    x[tid] = tsum;
}

// ===============================================================================================
// Unit sphere volume simulation functions
// ===============================================================================================
double HW1_SPHERE_host(int n)
{
    // Count of points inside the sphere
    float sum = 0;

    // Generate points and count those inside the sphere
    for (int i = 0; i < n; i++)
    {
        // Generate random points in the range [-1.0, 1.0]
        float x = 2.0f * RAND_UNIFORM - 1.0f;
        float y = 2.0f * RAND_UNIFORM - 1.0f;
        float z = 2.0f * RAND_UNIFORM - 1.0f;

        if (x * x + y * y + z * z <= 1.0f)
            sum++;
    }

    // Return the volume of the sphere
    return 8.0 * sum / n; // The volume of the unit cube is 8
}

double HW1_SPHERE_reduce1(int n, curandState *d_states)
{
    int threads = BLOCK_SIZE;
    int blocks = CEIL_DIV(n, threads);
    int blocks_reduce1 = n / threads;

    // Execute Monte Carlo simulation kernel
    float *d_counts;
    cudaMalloc(&d_counts, n * sizeof(double));

    do_MonteCarlo_simulation<<<blocks, threads, threads * sizeof(float)>>>(d_states, d_counts, n);

    // Reduce the simulation results
    // int blocks = n / threads;
    reduce1<<<blocks_reduce1, threads>>>(d_counts, n);
    reduce1<<<1, threads>>>(d_counts, blocks_reduce1 * threads);
    reduce1<<<1, 1>>>(d_counts, threads);
    cudaDeviceSynchronize();

    // Copy final result back to host
    float sum;
    cudaMemcpy(&sum, d_counts, sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(d_counts);

    // Return the volume of the sphere if (tid >= N)
    return 8.0 * sum / n; // The volume of the unit cube is 8
}

double HW1_SPHERE_thrust(int n, curandState *d_states)
{
    int threads = BLOCK_SIZE;
    int blocks = CEIL_DIV(n, threads);

    // Execute Monte Carlo simulation kernel
    float *d_counts;
    cudaMalloc(&d_counts, n * sizeof(double));

    do_MonteCarlo_simulation<<<blocks, threads, threads * sizeof(float)>>>(d_states, d_counts, n);

    // Reduce the simulation results
    thrust::device_vector<float> d_vec_counts(d_counts, d_counts + n);
    float sum = thrust::reduce(d_vec_counts.begin(), d_vec_counts.end());

    // Free allocated device memory
    cudaFree(d_counts);

    // Return the volume of the sphere
    return 8.0 * sum / n; // The volume of the unit cube is 8
}

void print_result(double unit_sphere_volume_exact, double unit_sphere_volume_simulated, float time)
{
    fprintf(stdout, "simulated = %.15f / ", unit_sphere_volume_simulated);
    fprintf(stdout, "exact = %.15f / ", unit_sphere_volume_exact);
    fprintf(stdout, "relative error = %.15f\n", fabs(unit_sphere_volume_simulated - unit_sphere_volume_exact) / unit_sphere_volume_exact);
    fprintf(stdout, "*** Time to estimate the volume of a sphere = %.3f(ms)\n", time);
}

int main()
{
    double unit_sphere_volume_exact = 4.0 / 3.0 * 3.14159265358979323846; // Exact volume of the sphere
    double unit_sphere_volume_simulted, unit_sphere_volume_simulated_host, unit_sphere_volume_simulated_reduce1, unit_sphere_volume_simulated_thrust;
    float total_time_ms, avg_time_ms_host, avg_time_ms_reduce1, avg_time_ms_thrust;

    // Initialize random states on host and device
    unsigned long seed = time(NULL);

    srand(seed);

    int threads = BLOCK_SIZE;
    int blocks = CEIL_DIV(N, threads);

    curandState *d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));

    init_curand_states<<<blocks, threads>>>(d_states, N, seed);
    cudaDeviceSynchronize();

    // ===============================================================================================
    // Compare functions
    // ===============================================================================================
    // HW1_SPHERE_host
    total_time_ms = 0;
    for (int i = 0; i < NUM_ITER_FOR_TEST; i++)
    {
        fprintf(stdout, "HW1_SPHERE_host Iter %d...", i);
        CHECK_TIME_START(_start);
        unit_sphere_volume_simulted = HW1_SPHERE_host(N);
        CHECK_TIME_END(_start, _end, _compute_time);
        total_time_ms += _compute_time;
        fprintf(stdout, "done\n");
    }
    avg_time_ms_host = total_time_ms / NUM_ITER_FOR_TEST;
    unit_sphere_volume_simulated_host = unit_sphere_volume_simulted;

    // HW1_SPHERE_reduce1
    total_time_ms = 0;
    for (int i = 0; i < NUM_ITER_FOR_TEST; i++)
    {
        fprintf(stdout, "HW1_SPHERE_reduce1 Iter %d...", i);
        CHECK_TIME_START(_start);
        unit_sphere_volume_simulted = HW1_SPHERE_reduce1(N, d_states);
        CHECK_TIME_END(_start, _end, _compute_time);
        total_time_ms += _compute_time;
        fprintf(stdout, "done\n");
    }
    avg_time_ms_reduce1 = total_time_ms / NUM_ITER_FOR_TEST;
    unit_sphere_volume_simulated_reduce1 = unit_sphere_volume_simulted;

    // HW1_SPHERE_thrust
    total_time_ms = 0;
    for (int i = 0; i < NUM_ITER_FOR_TEST; i++)
    {
        fprintf(stdout, "HW1_SPHERE_thrust Iter %d...", i);
        CHECK_TIME_START(_start);
        unit_sphere_volume_simulted = HW1_SPHERE_thrust(N, d_states);
        CHECK_TIME_END(_start, _end, _compute_time);
        total_time_ms += _compute_time;
        fprintf(stdout, "done\n");
    }
    avg_time_ms_thrust = total_time_ms / NUM_ITER_FOR_TEST;
    unit_sphere_volume_simulated_thrust = unit_sphere_volume_simulted;

    // Free device memory
    cudaFree(d_states);

    // Print results
    fprintf(stdout, "\nn = %d\n", N);

    fprintf(stdout, "\nHW1_SPHERE_host\nArea of unit sphere: ");
    print_result(unit_sphere_volume_exact, unit_sphere_volume_simulated_host, avg_time_ms_host);

    fprintf(stdout, "\nHW1_SPHERE_reduce1\nArea of unit sphere: ");
    print_result(unit_sphere_volume_exact, unit_sphere_volume_simulated_reduce1, avg_time_ms_reduce1);

    fprintf(stdout, "\nHW1_SPHERE_thrust\nArea of unit sphere: ");
    print_result(unit_sphere_volume_exact, unit_sphere_volume_simulated_thrust, avg_time_ms_thrust);

    // ===============================================================================================
    // Validate reduction methods
    // ===============================================================================================
    srand(0);

    float *rand_array = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        rand_array[i] = RAND_UNIFORM;
    }

    float sum_host = 0;
    for (int i = 0; i < N; i++)
    {
        sum_host += rand_array[i];
    }

    double sum_host_double = 0;
    for (int i = 0; i < N; i++)
    {
        sum_host_double += rand_array[i];
    }

    // reduce1
    float *d_rand_array;
    cudaMalloc(&d_rand_array, N * sizeof(float));
    cudaMemcpy(d_rand_array, rand_array, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks_reduce1 = N / threads;
    reduce1<<<blocks_reduce1, threads>>>(d_rand_array, N);
    reduce1<<<1, threads>>>(d_rand_array, blocks_reduce1 * threads);
    reduce1<<<1, 1>>>(d_rand_array, threads);
    cudaDeviceSynchronize();

    float sum_reduce1;
    cudaMemcpy(&sum_reduce1, d_rand_array, sizeof(float), cudaMemcpyDeviceToHost);

    // Thrust
    cudaMemcpy(d_rand_array, rand_array, N * sizeof(float), cudaMemcpyHostToDevice);
    thrust::device_vector<float> d_vec_counts(d_rand_array, d_rand_array + N);
    float sum_thrust = thrust::reduce(d_vec_counts.begin(), d_vec_counts.end());

    cudaFree(d_rand_array);

    fprintf(stdout, "\nValidate reduction methods\n");
    fprintf(stdout, "Host(float): %.15f\n", sum_host);
    fprintf(stdout, "Host(double): %.15f\n", sum_host_double);
    fprintf(stdout, "Reduce1: %.15f\n", sum_reduce1);
    fprintf(stdout, "Thrust: %.15f\n", sum_thrust);
}
