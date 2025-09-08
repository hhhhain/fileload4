// serialize
for (int i = 0; i < mKernelCount; ++i)
    write(d, mYoloKernel[i]);

// deserialize
mYoloKernel.resize(mKernelCount);
for (int i = 0; i < mKernelCount; ++i)
    read(d, mYoloKernel[i]);
