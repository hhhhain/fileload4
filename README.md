反序列号之后我又一次读了，发现结果还是正确的。
mThreadCount is 256
mMaxOutObject is 1000
mKernelCount is 24
mYoloV5NetWidth is 640
mYoloV5NetHeight is 1088
mClassCount is 26

反序列化的代码：
IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call YoloLayerPlugin::destroy()
  YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}
会不会跟setPluginNamespace有关？

接下来是执行enqueue吗？
int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
  return 0;
}
然后我在forwardgpu里面打印就不对了，但是我发现了一些规律，有些错 有些对：
mThreadCount is 0
mMaxOutObject is 1000
numElem is 0
mYoloV5NetWidth is 640
mYoloV5NetHeight is 1088
yolo.width is 1117782016
yolo.height is 1124597760
mClassCount is 26
outputElem is 38001
