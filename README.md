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
    // push_back用来在vector末尾添加一个元素.
    kernels.push_back(kernel);
  }
  plugin_fields[1].data = &kernels[0];
  plugin_fields[1].length = kernels.size();
  plugin_fields[1].name = "kernels";
  plugin_fields[1].type = PluginFieldType::kFLOAT32;
  PluginFieldCollection plugin_data;
  plugin_data.nbFields = 2;
  plugin_data.fields = plugin_fields;
  
  // 这里创建了具体的plugin实例.
  IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
  std::vector<ITensor*> input_tensors;
  // 输入的定义,std::vector<IConvolutionLayer*> dets, dets是一个vector容器,容器里面每个元素是IConvolutionLayer*,指向三个输出卷积层的指针.为什么知道是输出的三个卷积层呢,这个函数被调用的地方有dets的具体赋值.
  // 遍历dets,也就是每个输出层,一共3个
  for (auto det: dets) {
    // 获取输出卷积层的输出tensor,放进input_tensors.为什么叫input呢,因为是plugin的input. 必须要索引,不能().
    input_tensors.push_back(det->getOutput(0));
  }
  // 调用addPluginV2把plugin_obj plugin实例插入网络.
  // addPluginV2是tensorrt内置的API, 看不到实现.
  auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
  return yolo;
}




    concat_out = network.get_layer(293).get_output(0)
    network.unmark_output(concat_out)

    plugin_layer = network.add_plugin_v2([concat_out], plugin)
    plugin_layer.get_output(0).name = "out_original"
    plugin_layer.get_output(1).name = "out_plus_one"

    network.mark_output(plugin_layer.get_output(0))
    network.mark_output(plugin_layer.get_output(1))
