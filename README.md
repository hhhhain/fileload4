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
}

// 假设这是你老版 plugin 的 enqueue
int YoloLayerPlugin::enqueue(int batchSize,
                             const void* const* inputs,
                             void* TRT_CONST_ENQUEUE* outputs,
                             void* workspace,
                             cudaStream_t stream) TRT_NOEXCEPT
{
    // 打印 debug：只打印第0个输入的前10个元素
    // 使用你的风格 cast 成 const float* const*
    const float* d_input = ((const float* const*)inputs)[0];

    // 计算要打印的长度 n，可以根据实际输入长度调整
    int n = batchSize * mMaxOutObject; // 或者你真实输入长度

    // 调用 debug kernel
    Tn::debugKernel<<<1, 1, 0, stream>>>(d_input, n);

    // 确保 printf 输出完成
    cudaStreamSynchronize(stream);
