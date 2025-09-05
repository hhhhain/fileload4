static IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets, bool is_segmentation = false) {
  auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
  auto anchors = getAnchors(weightMap, lname);
  PluginField plugin_fields[2];
  int netinfo[5] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox, (int)is_segmentation};
  plugin_fields[0].data = netinfo;
  plugin_fields[0].length = 5;
  plugin_fields[0].name = "netinfo";
  plugin_fields[0].type = PluginFieldType::kFLOAT32;

  //load strides from Detect layer
  assert(weightMap.find(lname + ".strides") != weightMap.end() && "Not found `strides`, please check gen_wts.py!!!");
  Weights strides = weightMap[lname + ".strides"];
  auto *p = (const float*)(strides.values);
  std::vector<int> scales(p, p + strides.count);

  std::vector<YoloKernel> kernels;
  for (size_t i = 0; i < anchors.size(); i++) {
    YoloKernel kernel;
    kernel.width = kInputW / scales[i];
    kernel.height = kInputH / scales[i];
    memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
    kernels.push_back(kernel);
  }
  plugin_fields[1].data = &kernels[0];
  plugin_fields[1].length = kernels.size();
  plugin_fields[1].name = "kernels";
  plugin_fields[1].type = PluginFieldType::kFLOAT32;
  PluginFieldCollection plugin_data;
  plugin_data.nbFields = 2;
  plugin_data.fields = plugin_fields;
  IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
  std::vector<ITensor*> input_tensors;
  for (auto det: dets) {
    input_tensors.push_back(det->getOutput(0));
  }
  auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
  return yolo;
}
