  int netinfo[5] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox, (int)is_segmentation};
  plugin_fields[0].data = netinfo;
  plugin_fields[0].length = 5;
  plugin_fields[0].name = "netinfo";
  plugin_fields[0].type = PluginFieldType::kFLOAT32;


  YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
  using namespace Tn; // 使用命名空间Tn下的函数，write和read应该就是。
  const char *d = reinterpret_cast<const char *>(data), *a = d; // 把void传入的data转换类型，命名为d，同时把a设为d的拷贝。
  read(d, mClassCount); // 从d开始读，并移动指针，所以下面都是从d开始读的，因为指针跟着变了。里面的内容是创建的时候写入的。这些都是plugin的成员变量，会在后面的时候使用。访问成员变量来用。
  read(d, mThreadCount);
  read(d, mKernelCount);
  read(d, mYoloV5NetWidth);
  read(d, mYoloV5NetHeight);
  read(d, mMaxOutObject);
  read(d, is_segmentation_);
