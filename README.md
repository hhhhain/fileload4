const float* d_input = ((const float* const*)inputs)[0];
  Tn::debugKernel<<<1, 1, 0, stream>>>(d_input, n);
__global__ void debugKernel(const T* data, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)  // 只用第一个线程打印
    {
        for (int i = 0; i < n && i < 10; ++i)
        {
            printf("data[%d] = %f\n", i, static_cast<float>(data[i]));
        }
    }
}

const __half* const* d_input_half = (const __half* const*)inputs;
Tn::debugKernel_half<<<1, 1, 0, stream>>>(d_input_half, n);
