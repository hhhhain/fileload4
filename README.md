void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {  
  using namespace Tn;
  char* d = static_cast<char*>(buffer), *a = d;
  write(d, mClassCount);
  write(d, mThreadCount);
  write(d, mKernelCount); // 比如mKernelCount为3，有3个头    1
  write(d, mYoloV5NetWidth);
  write(d, mYoloV5NetHeight);
  write(d, mMaxOutObject);
  write(d, is_segmentation_);
  auto kernelSize = mKernelCount * sizeof(YoloKernel);
  memcpy(d, mYoloKernel.data(), kernelSize);
  d += kernelSize; // 同理。

  assert(d == a + getSerializationSize()); // 同理，用下面的函数计算，是否相等。


  printf("mThreadCount is %d\n", mThreadCount);    
  printf("mMaxOutObject is %d\n", mMaxOutObject);  
  printf("mKernelCount is %d\n", mKernelCount);
  printf("mYoloV5NetWidth is %d\n", mYoloV5NetWidth);  
  printf("mYoloV5NetHeight is %d\n", mYoloV5NetHeight);   
  printf("mClassCount is %d\n", mClassCount);     
  exit(0);
我得到的输出：
  mThreadCount is 256
mMaxOutObject is 1000
mKernelCount is 24
mYoloV5NetWidth is 640
mYoloV5NetHeight is 1088
mClassCount is 26
