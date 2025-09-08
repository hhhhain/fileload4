nvinfer1::Dims YoloLayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                                    int nbInputDims) TRT_NOEXCEPT {
    int total_size = mMaxOutObject * sizeof(Detection) / sizeof(float);
    return nvinfer1::Dims3(total_size + 1, 1, 1);
}
