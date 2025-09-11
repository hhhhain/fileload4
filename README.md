    // 清零每张图的 res_count
    for (int b = 0; b < batchSize; ++b) {
        float* res_count = static_cast<float*>(outputs[0]) + b * outputElem;
        cudaMemsetAsync(res_count, 0, sizeof(float), stream);
    }
