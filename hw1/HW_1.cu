#include <stdio.h>

#define rand_uniform ((float)rand() / RAND_MAX)

float HW1_SPHERE_host(int n);

int main()
{
    int N = 1 << 24;                                                      // Number of points
    double unit_sphere_volume_exact = 4.0 / 3.0 * 3.14159265358979323846; // Exact volume of the sphere
    double unit_sphere_volume_simulated;

    unit_sphere_volume_simulated = HW1_SPHERE_host(N);
    fprintf(stdout, "\nArea of unit sphere: ");
    fprintf(stdout, "simulated = %.15f / ", unit_sphere_volume_simulated);
    fprintf(stdout, "exact = %.15f / ", unit_sphere_volume_exact);
    fprintf(stdout, "relative error = %.15f\n", unit_sphere_volume_simulated);
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