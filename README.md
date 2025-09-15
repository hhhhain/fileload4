YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, bool is_segmentation, const std::vector<YoloKernel>& vYoloKernel) {
  mClassCount = classCount; 
  mYoloV5NetWidth = netWidth; 
  mYoloV5NetHeight = netHeight; 
  mMaxOutObject = maxOut;
  is_segmentation_ = is_segmentation;
  mYoloKernel = vYoloKernel; 
  mKernelCount = vYoloKernel.size();

  那这里的 vYoloKernel.size() 怎么理解呢？我以为应该查看YoloKernel，我去看YoloKernel的定义就是我说的这个结构体
