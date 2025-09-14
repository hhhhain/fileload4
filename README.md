我的做法是：
det1 = network.get_layer(242).get_output(0)
det2 = network.get_layer(267).get_output(0)
det3 = network.get_layer(292).get_output(0)

add_yolo_layer_py(network, det_tensors=[det1,det2,det3], concat_layer_index=293, is_segmentation=False)
这det1到det3相当于是这242 267 292的输出值对吧？相当于是yolov5s的3个detection层输出。我把这个输出再送进plugin去做框筛选。

另外一种c++的实现，目的跟我是一样的：
IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), kNumAnchor * (kNumClass + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
  for (auto det: dets) {
    // 获取输出卷积层的输出tensor,放进input_tensors.为什么叫input呢,因为是plugin的input. 必须要索引,不能().
    input_tensors.push_back(det->getOutput(0));
  }
  // 调用addPluginV2把plugin_obj plugin实例插入网络.
  // addPluginV2是tensorrt内置的API, 看不到实现.
  auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);

  我们两个的写法是对应的吗？我现在打印plugin的输入，发现这两个写法的plugin输入值不一样，怀疑是写法问题。
