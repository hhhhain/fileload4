det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), kNumAnchor * (kNumClass + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
