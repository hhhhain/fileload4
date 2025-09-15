YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, bool is_segmentation, const std::vector<YoloKernel>& vYoloKernel) {
  mClassCount = classCount;
  mYoloV5NetWidth = netWidth; 
  mYoloV5NetHeight = netHeight;
  mMaxOutObject = maxOut;
  is_segmentation_ = is_segmentation;
  mYoloKernel = vYoloKernel; 
  mKernelCount = vYoloKernel.size(); 

  CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*))); 
  size_t AnchorLen = sizeof(float)* kNumAnchor * 2; 
  for (int ii = 0; ii < mKernelCount; ii++) { 
    CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen)); 
    const auto& yolo = mYoloKernel[ii];
    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice)); 
  }
}
