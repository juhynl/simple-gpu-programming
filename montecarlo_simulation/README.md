# Montecarlo Simulation with CUDA
Let's simulate the volume of a sphere with a radius of 1 (which is $\frac{4}{3}\pi$) by generating uniform random numbers on a GPU.
- Generate $n$ points within the space $ [-1.0, 1.0] \times [-1.0, 1.0] \times [-1.0, 1.0] $ for sufficiently large $n$.
- Count the number of points that fall inside the sphere and estimate the sphere's volume.

## Methods
**[Method 1]** Implement the following function that operates on the host (CPU):
```cpp
float HW1_SPHERE_host(int n);
```
**[Method 2]** Extend the reduce1 kernel method explained during class to implement the function:
```cpp
float HW1_SPHERE_reduce1(int n);
```
**[Method 3]** Appropriately use CUDA's thrust library functions to implement the function. Refer to [NVIDIA's thrust documentation](https://nvidia.github.io/cccl/thrust/) and [CUDA documentation](https://docs.nvidia.com/cuda/) for relevant details:
```cpp
float HW1_SPHERE_thrust(int n);
```
