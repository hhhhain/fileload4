[09/02/2025-12:43:58] [E] [TRT] ModelImporter.cpp:771: While parsing node number 417 [QuantizeLinear -> "/model.9/cv2/conv/_weight_quantizer/QuantizeLinear_output_0"]:
[09/02/2025-12:43:58] [E] [TRT] ModelImporter.cpp:772: --- Begin node ---
[09/02/2025-12:43:58] [E] [TRT] ModelImporter.cpp:773: input: "model.9.cv2.conv.weight"
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

[09/02/2025-12:43:58] [E] [TRT] ModelImporter.cpp:774: --- End node ---
[09/02/2025-12:43:58] [E] [TRT] ModelImporter.cpp:777: ERROR: builtin_op_importers.cpp:1197 In function QuantDequantLinearHelper:
[6] Assertion failed: scaleAllPositive && "Scale coefficients must all be positive"
[09/02/2025-12:43:58] [E] Failed to parse onnx file
[09/02/2025-12:43:58] [I] Finished parsing network model. Parse time: 0.230179
[09/02/2025-12:43:58] [E] Parsing model failed
[09/02/2025-12:43:58] [E] Failed to create engine from model or file.
[09/02/2025-12:43:58] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v8601] # trtexec --onnx=/home/ma-user/work/copy/yolov5_QAT-main/qat.onnx --saveEngine=/home/ma-user/work/copy/files/video-deal-search/video-deal-service/weights/CP26classes_epoch_180_int8_bs10_640_1088_from_exec.trt --int8 --workspace=1024000 --verbose







import onnx
import numpy as np

model = onnx.load("qat.onnx")
for node in model.graph.node:
    if node.name.endswith("conv/_weight_quantizer/QuantizeLinear"):
        print(node)
for init in model.graph.initializer:
    if "conv/_weight_quantizer/Constant_output_0" in init.name:
        arr = onnx.numpy_helper.to_array(init)
        print("scale:", arr.min(), arr.max(), arr[:10])









import onnx_graphsurgeon as gs
import numpy as np

graph = gs.import_onnx(onnx.load("qat.onnx"))
for tensor in graph.tensors().values():
    if "conv/_weight_quantizer/Constant_output_0" in tensor.name:
        arr = np.abs(tensor.values)
        arr[arr == 0] = 1e-8
        tensor.values = arr
onnx.save(gs.export_onnx(graph), "qat_fixed.onnx")








def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load

        # model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

        # Modified by maggie.
        # 1. Since we benchmark the speed using TensorRT backend, so it is not necesary to fuse.
        # 2. If fuse, the fuse_conv_and_bn function will be called, then the quant_nn.QuantConv2d will be replace by noraml Conv2d
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
