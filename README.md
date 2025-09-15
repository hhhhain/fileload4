// 假设 vYoloKernel 已经有 96 个元素，但我们只要前 3 个
size_t valid_count = 3;  // 你真正需要的 kernel 数量

if (vYoloKernel.size() > valid_count) {
    // 保留前 valid_count 个
    std::vector<YoloKernel> tmp(vYoloKernel.begin(), vYoloKernel.begin() + valid_count);
    vYoloKernel = tmp;  // 覆盖原来的 vector
}

// 之后你可以继续使用 vYoloKernel，它现在只有 3 个 head
mKernelCount = vYoloKernel.size();  // 3
