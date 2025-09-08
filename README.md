static IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets, bool is_segmentation = false) {
  auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
  auto anchors = getAnchors(weightMap, lname);
  PluginField plugin_fields[2];

  // kNumClass, kInputW, kInputH, kMaxNumOutputBbox这几个参数在config.h头文件里宏定义.
  int netinfo[5] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox, (int)is_segmentation};
  plugin_fields[0].data = netinfo;
  plugin_fields[0].length = 5;
  plugin_fields[0].name = "netinfo";
  plugin_fields[0].type = PluginFieldType::kFLOAT32;
