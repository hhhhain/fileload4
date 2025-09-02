    (9): SPPF(
      (cv1): Conv(
        (conv): QuantConv2d(
          512, 256, kernel_size=(1, 1), stride=(1, 1)
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=8.3423 calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.4133, 1.2761](256) calibrator=MaxCalibrator scale=1.0 quant)
        )
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): QuantConv2d(
          1024, 512, kernel_size=(1, 1), stride=(1, 1)
          (_input_quantizer): TensorQuantizer(8bit fake per-tensor amax=21.2856 calibrator=HistogramCalibrator scale=1.0 quant)
          (_weight_quantizer): TensorQuantizer(8bit fake axis=0 amax=[0.0000, 0.3353](512) calibrator=MaxCalibrator scale=1.0 quant)
        )
        (act): SiLU(inplace=True)
      )
      (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    )
