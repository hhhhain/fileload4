 private:
  void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
  int mThreadCount = 256;
  const char* mPluginNamespace;
  int mKernelCount;
  int mClassCount;
  int mYoloV5NetWidth;
  int mYoloV5NetHeight;
  int mMaxOutObject;
  bool is_segmentation_;
  std::vector<YoloKernel> mYoloKernel;
  void** mAnchor;
};

我发现在yoloyaler.h里面已经有了初始值了呀，为什么printf是0呢？不会是我没用这个头文件？
