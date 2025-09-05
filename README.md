  IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), kNumAnchor * (kNumClass + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
  auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
  ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
  auto cat19 = network->addConcatenation(inputTensors19, 2);
  auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
  IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), kNumAnchor * (kNumClass + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
  auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
  ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
  auto cat22 = network->addConcatenation(inputTensors22, 2);
  auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
  IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), kNumAnchor * (kNumClass + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

  auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
  yolo->getOutput(0)->setName(kOutputTensorName);
  network->markOutput(*yolo->getOutput(0));
