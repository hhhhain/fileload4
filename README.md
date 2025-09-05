  auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
  // 这里把yolo层,也就是plugin的处理结果获取到,并给他取一个名字,叫做kOutputTensorName
  yolo->getOutput(0)->setName(kOutputTensorName);
  // tensorrt要求你必须显示告诉他哪几个tensor是最终输出,markOutput就是起这个作用. 这里把*yolo->getOutput(0)标记为network的最终输出
  network->markOutput(*yolo->getOutput(0));
