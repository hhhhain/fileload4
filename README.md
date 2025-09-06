import tensorrt as trt
import numpy as np

def add_yolo_layer(network, weight_map, lname, dets, is_segmentation=False):
    # 1. 获取插件 creator
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("YoloLayer_TRT", "1")
    if creator is None:
        raise RuntimeError("YoloLayer_TRT plugin not found in registry!")

    # 2. 构造 netinfo
    netinfo = np.array(
        [kNumClass, kInputW, kInputH, kMaxNumOutputBbox, int(is_segmentation)],
        dtype=np.float32
    )
    netinfo_field = trt.PluginField(
        name="netinfo",
        data=netinfo,
        type=trt.PluginFieldType.FLOAT32
    )

    # 3. 加载 strides
    if lname + ".strides" not in weight_map:
        raise RuntimeError("Not found strides, please check gen_wts.py!!!")
    strides = weight_map[lname + ".strides"]  # 这里通常是 Weights
    scales = np.array(strides, dtype=np.float32)  # strides.values -> numpy

    # 4. 构造 kernels
    kernels = []
    anchors = getAnchors(weight_map, lname)  # 需要你实现 getAnchors()
    for i in range(len(anchors)):
        w = kInputW / scales[i]
        h = kInputH / scales[i]
        # 一个 YoloKernel = (width, height, anchors)
        kernel = [w, h] + anchors[i]
        kernels.extend(kernel)
    kernels = np.array(kernels, dtype=np.float32)

    kernels_field = trt.PluginField(
        name="kernels",
        data=kernels,
        type=trt.PluginFieldType.FLOAT32
    )

    # 5. 封装 PluginFieldCollection
    field_collection = trt.PluginFieldCollection([netinfo_field, kernels_field])

    # 6. 创建 plugin
    plugin_obj = creator.create_plugin("yololayer", field_collection)

    # 7. 收集 det 层输出作为输入
    input_tensors = [det.get_output(0) for det in dets]

    # 8. 插入 plugin
    yolo_layer = network.add_plugin_v2(inputs=input_tensors, plugin=plugin_obj)
    return yolo_layer
