    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        // compute number of elements in input tensor
        int volume = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; ++i) volume *= inputDesc[0].dims.d[i];

        // copy input -> outputs[0] (device to device)
        cudaError_t err = cudaMemcpyAsync(outputs[0], inputs[0], (size_t)volume * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) return -1;

        // launch kernel to compute outputs[1] = input + 1
        const int threads = 256;
        const int blocks = (volume + threads - 1) / threads;
        addOneKernel<<<blocks, threads, 0, stream>>>((const float*)inputs[0], (float*)outputs[1], volume);
        // optional: check kernel launch error (can't return cudaError_t directly)
        err = cudaGetLastError();
        if (err != cudaSuccess) return -2;

        return 0;
    }

    // IPluginV2Ext / IPluginV2DynamicExt required methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        return inputTypes[0];
    }
