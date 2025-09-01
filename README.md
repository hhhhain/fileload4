ModelImporter.cpp:771: While parsing node number 417 [QuantizeLinear -> "/model.9/cv2/conv/_weight_quantizer/QuantizeLinear_output_0"]:
[09/01/2025-21:22:14] [E] [TRT] ModelImporter.cpp:772: --- Begin node ---
[09/01/2025-21:22:14] [E] [TRT] ModelImporter.cpp:773: input: "model.9.cv2.conv.weight"
input: "/model.9/cv2/conv/_weight_quantizer/Constant_output_0"
input: "/model.9/cv2/conv/_weight_quantizer/Constant_1_output_0"
output: "/model.9/cv2/conv/_weight_quantizer/QuantizeLinear_output_0"
name: "/model.9/cv2/conv/_weight_quantizer/QuantizeLinear"
op_type: "QuantizeLinear"
attribute {
  name: "axis"
  i: 0
  type: INT
}

[09/01/2025-21:22:14] [E] [TRT] ModelImporter.cpp:774: --- End node ---
[09/01/2025-21:22:14] [E] [TRT] ModelImporter.cpp:777: ERROR: builtin_op_importers.cpp:1197 In function QuantDequantLinearHelper:
[6] Assertion failed: scaleAllPositive && "Scale coefficients must all be positive"
[09/01/2025-21:22:14] [E] Failed to parse onnx file
[09/01/2025-21:22:14] [I] Finished parsing network model. Parse time: 0.241168
[09/01/2025-21:22:14] [E] Parsing model failed
[09/01/2025-21:22:14] [E] Failed to create engine from model or file.
[09/01/2025-21:22:14] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v8601] # trtexec --onnx=/home/ma-user/work/copy/yolov5_QAT-main/qat.onnx --saveEngine=/home/ma-user/work/copy/files/video-deal-search/video-deal-service/weights/CP26classes_epoch_180_int8_bs10_640_1088_from_exec.trt --int8 --workspace=1024000 --verbose
