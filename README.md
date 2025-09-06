host output: [[[7.562e+00, 6.125e+00, 1.780e+01, ..., 7.007e-02,

host output: [[[5.54687500e+00 5.03125000e+00 1.57578125e+01 ...

__global__ void addOneKernel(const float* input, float* output1, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output1[idx] = input[idx] + 1.0f;
}
