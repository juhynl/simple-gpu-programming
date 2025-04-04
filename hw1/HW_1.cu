#include <stdio.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#define rand_uniform ((float)rand() / RAND_MAX)

float HW1_SPHERE_host(int n);
float HW1_SPHERE_reduce1(int n);
float HW1_SPHERE_thrust(int n);

__global__ void init_curand_states(curandState *states, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // each thread gets a different state
    curand_init(seed, idx, 0, &states[idx]);
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

__global__ void reduce1(float *x, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float tsum = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int k = tid; k < N; k += stride)
    {
        tsum += x[k];
    }
    x[tid] = tsum;
}

int main()
{
    int N = 1 << 24;                                                      // Number of points
    double unit_sphere_volume_exact = 4.0 / 3.0 * 3.14159265358979323846; // Exact volume of the sphere
    double unit_sphere_volume_simulated;

    unit_sphere_volume_simulated = HW1_SPHERE_host(N);

    fprintf(stdout, "\nHW1_SPHERE_host\nArea of unit sphere: ");
    fprintf(stdout, "simulated = %.15f / ", unit_sphere_volume_simulated);
    fprintf(stdout, "exact = %.15f / ", unit_sphere_volume_exact);
    fprintf(stdout, "relative error = %.15f\n", fabs(unit_sphere_volume_simulated - unit_sphere_volume_exact) / unit_sphere_volume_exact);

    unit_sphere_volume_simulated = HW1_SPHERE_reduce1(N);
    fprintf(stdout, "\nHW1_SPHERE_reduce1\nArea of unit sphere: ");
    fprintf(stdout, "simulated = %.15f / ", unit_sphere_volume_simulated);
    fprintf(stdout, "exact = %.15f / ", unit_sphere_volume_exact);
    fprintf(stdout, "relative error = %.15f\n", fabs(unit_sphere_volume_simulated - unit_sphere_volume_exact) / unit_sphere_volume_exact);

    unit_sphere_volume_simulated = HW1_SPHERE_thrust(N);
    fprintf(stdout, "\nHW1_SPHERE_reduce1\nArea of unit sphere: ");
    fprintf(stdout, "simulated = %.15f / ", unit_sphere_volume_simulated);
    fprintf(stdout, "exact = %.15f / ", unit_sphere_volume_exact);
    fprintf(stdout, "relative error = %.15f\n", fabs(unit_sphere_volume_simulated - unit_sphere_volume_exact) / unit_sphere_volume_exact);
}

float HW1_SPHERE_host(int n)
{
    // Initialize random generator
    srand(time(NULL));

    // Count of points inside the sphere
    int count = 0;

    // Generate points and count those inside the sphere
    for (int i = 0; i < n; i++)
    {
        // Generate random points in the range [-1.0, 1.0]
        float x = 2.0f * rand_uniform - 1.0f;
        float y = 2.0f * rand_uniform - 1.0f;
        float z = 2.0f * rand_uniform - 1.0f;

        if (x * x + y * y + z * z <= 1.0f)
            count++;
    }

    // Return the volume of the sphere
    return 8.0 * count / n; // The volume of the unit cube is 8
}

float HW1_SPHERE_reduce1(int n)
{
    int threads = 256;
    int blocks = n / threads; // assume N is a multiple of thread block size.

    curandState *d_states;
    cudaMalloc(&d_states, n * sizeof(curandState));

    init_curand_states<<<blocks, threads>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    float *d_counts;
    cudaMalloc(&d_counts, n * sizeof(double));

    do_MonteCarlo_simulation<<<blocks, threads, threads * sizeof(float)>>>(d_states, d_counts, n);
    cudaDeviceSynchronize();

    // free(counts);

    // do reduction here
    reduce1<<<blocks, threads>>>(d_counts, n);
    reduce1<<<1, threads>>>(d_counts, blocks * threads);
    reduce1<<<1, 1>>>(d_counts, threads);

    float sum;
    cudaMemcpy(&sum, d_counts, sizeof(float), cudaMemcpyDeviceToHost);

    // Return the volume of the sphere
    return 8.0 * sum / n; // The volume of the unit cube is 8
}

float HW1_SPHERE_thrust(int n)
{
    int threads = 256;
    int blocks = n / threads; // assume N is a multiple of thread block size.

    curandState *d_states;
    cudaMalloc(&d_states, n * sizeof(curandState));

    init_curand_states<<<blocks, threads>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    float *d_counts;
    cudaMalloc(&d_counts, n * sizeof(double));

    do_MonteCarlo_simulation<<<blocks, threads, threads * sizeof(float)>>>(d_states, d_counts, n);
    cudaDeviceSynchronize();

    thrust::device_vector<float> d_vec_counts(d_counts, d_counts + n);
    float sum = thrust::reduce(d_vec_counts.begin(), d_vec_counts.end());

    // Return the volume of the sphere
    return 8.0 * sum / n; // The volume of the unit cube is 8
}