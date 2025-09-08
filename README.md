void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {  
  using namespace Tn;
  char* d = static_cast<char*>(buffer), *a = d;
  write(d, mClassCount);
  write(d, mThreadCount);
  write(d, mKernelCount); // 比如mKernelCount为3，有3个头
  write(d, mYoloV5NetWidth);
  write(d, mYoloV5NetHeight);
  write(d, mMaxOutObject);
  write(d, is_segmentation_);
  auto kernelSize = mKernelCount * sizeof(YoloKernel);
  memcpy(d, mYoloKernel.data(), kernelSize);
  d += kernelSize; // 同理。

  assert(d == a + getSerializationSize()); // 同理，用下面的函数计算，是否相等。
}


IHostMemory* serialized_engine = engine->serialize();
