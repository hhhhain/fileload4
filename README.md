int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {

  int numInputDims = getNbInputs();  // 输入 tensor 个数
for (int i = 0; i < numInputDims; i++) {
    auto dim = getInputDimensions(i);
    auto dtype = getInputDataType(i);  // 打印数据类型（构建时接口）

    std::cout << "Input " << i << " dims: ";
    for (int j = 0; j < dim.nbDims; j++) {
        std::cout << dim.d[j] << " ";
    }
    std::cout << " | dtype = " << static_cast<int>(dtype) << std::endl;
}

// 打印完成后直接退出进程
std::cout << "Plugin debug info done, exiting..." << std::endl;
exit(0);
  
  
  
  
  forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
  return 0;
}
