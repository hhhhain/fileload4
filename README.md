size_t valid_count = 3;
if (vYoloKernel.size() > valid_count) {
    mYoloKernel.assign(vYoloKernel.begin(), vYoloKernel.begin() + valid_count);
} else {
    mYoloKernel = vYoloKernel;
}
mKernelCount = mYoloKernel.size();
