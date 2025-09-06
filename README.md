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











import numpy as np
import tensorrt as trt
import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def add_yolo_layer_py(network, weight_map, lname, det_tensors=None, concat_layer_index=None, is_segmentation=False):
    """
    - network: trt.INetworkDefinition
    - weight_map: 你的权重字典（用于 getAnchors/get strides）
    - lname: e.g. "model.24"
    - det_tensors: 列表 of ITensor (三个 detection head 的输出), 优先使用这个
    - concat_layer_index: 如果你想直接处理 concat 层，传该 layer index（int）
    - 返回: 插入的 plugin layer (ILayer)
    """
    # 1) 确保插件库被加载并初始化（在外部只需做一次也可以）
    # ctypes.CDLL("/path/to/libYoloPlugin.so", mode=ctypes.RTLD_GLOBAL)
    # trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    # 2) 获取 Creator
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("YoloLayer_TRT", "1", "")
    if creator is None:
        raise RuntimeError("YoloLayer_TRT plugin not found in registry! Did you load the plugin .so and call init_libnvinfer_plugins?")

    # 3) 准备 netinfo
    # 注意：C++ createPlugin 里通常会把字段 reinterpret_cast 成 int* 或 float*
    # 根据你 plugin 的实现把类型对齐——这里用 int32（C++ 中通常读取为 int）
    netinfo = np.array([kNumClass, kInputW, kInputH, kMaxNumOutputBbox, int(is_segmentation)], dtype=np.int32)
    f_netinfo = trt.PluginField("netinfo", netinfo, trt.PluginFieldType.INT32)

    # 4) 从 weight_map 读取 strides / anchors（示例 getAnchors 函数需你自己实现）
    if (lname + ".strides") not in weight_map:
        raise RuntimeError("Not found `strides`, please check gen_wts.py!!!")
    # strides 权重在你的代码里可能是一个 Weights 对象：这里假设 weight_map[...] 可以转为 numpy
    strides_weights = weight_map[lname + ".strides"]  # 形式视你的 weight_map 实现而定
    # 如果 strides_weights 是 Tensorrt Weights（.values/.count），你需要把它转成 numpy：
    # e.g. strides = np.frombuffer(strides_weights.values, dtype=np.float32, count=strides_weights.count)
    # 这里假设你把它换成了 python list/np.array:
    p = np.array(strides_weights, dtype=np.float32)  # <-- adapt as needed
    scales = p.astype(np.int32)  # strides 例如 [8,16,32] -> scales

    # 5) 构造 kernels 内容 —— 必须与 C++ 插件期望的内存布局一致
    # 在 C++ 里 YoloKernel 结构体可能像: struct YoloKernel { int width; int height; float anchors[6]; }
    # 因此在 Python 需要按相同字节序打包 (int32, int32, float32*x)
    anchors = getAnchors(weight_map, lname)  # 例如返回 [[a0,a1,a2,a3,a4,a5], [...], [...]]
    num_kernels = len(anchors)
    anchors_per = len(anchors[0])  # 常见为 6 (3 anchor pairs)
    # 定义 numpy 结构化 dtype 与 C++ struct 对齐
    dtype_kernel = np.dtype([('width','i4'), ('height','i4'), ('anchors', 'f4', anchors_per)])
    kernels_arr = np.zeros((num_kernels,), dtype=dtype_kernel)
    for i in range(num_kernels):
        kernels_arr[i]['width'] = int(kInputW // int(scales[i]))   # 与C++一致
        kernels_arr[i]['height'] = int(kInputH // int(scales[i]))
        kernels_arr[i]['anchors'] = np.array(anchors[i], dtype=np.float32)

    # PluginField 要传 raw bytes 或 numpy array 的 view；C++ 端会 reinterpret_cast((YoloKernel*)fields.data)
    # 因此把结构数组 view 为 float32 bytes 或直接使用 kernels_arr.tobytes()
    # trt.PluginField 在 python 端通常接受 numpy array directly
    kernels_raw = np.frombuffer(kernels_arr.tobytes(), dtype=np.uint8)  # bytes view
    # 但 PluginFieldType 必须反映数据内容类型；很多插件在 C++ 里期望 FLOAT32 数组，所以我们也传FLOAT32视图
    # 这里把 struct bytes 以 float32 的视角传递（C++ 插件如果直接 memcpy sizeof(YoloKernel) 会正确）
    kernels_field = trt.PluginField("kernels", np.array(list(kernels_arr.tobytes()), dtype=np.uint8), trt.PluginFieldType.DR_INT8) 
    # ——上面是通用方法（把 bytes 作为 int8 / uint8 传入），插件侧需要按字节解析。
    # 注意：某些 plugin 期望 PluginFieldType.FLOAT32 并且 fields.data 指向 float32 数组，
    # 如果你的 C++ 插件是把 kernels 当作 float buffer 解读（而不是 struct），需要相应修改为
    # kernels_flat = np.array([], dtype=np.float32)  # fill accordingly
    # kernels_field = trt.PluginField("kernels", kernels_flat, trt.PluginFieldType.FLOAT32)

    # 6) 组装 PluginFieldCollection
    fields = [f_netinfo, kernels_field]
    field_collection = trt.PluginFieldCollection(fields)

    # 7) 创建 plugin 实例
    plugin_obj = creator.create_plugin(name="yololayer", field_collection=field_collection)
    if plugin_obj is None:
        raise RuntimeError("creator.create_plugin returned None")

    # 8) 选择输入 tensors：优先使用 det_tensors，否则用 concat_layer_index
    if det_tensors is not None:
        inputs = det_tensors
    elif concat_layer_index is not None:
        concat_tensor = network.get_layer(concat_layer_index).get_output(0)
        # 如果 concat_tensor 在 network outputs 中已经被标记 output，需要先 unmark
        try:
            network.unmark_output(concat_tensor)
        except Exception:
            # 如果没被标记，这会抛错或返回；忽略
            pass
        inputs = [concat_tensor]
    else:
        raise ValueError("Either det_tensors or concat_layer_index must be provided")

    # 9) 把 plugin 插入网络
    yolo_layer = network.add_plugin_v2(inputs=inputs, plugin=plugin_obj)
    if yolo_layer is None:
        raise RuntimeError("network.add_plugin_v2 returned None")

    # 10) 命名并标记输出（示例）
    yolo_layer.get_output(0).name = "yolo_out_original"
    yolo_layer.get_output(1).name = "yolo_out_post"
    network.mark_output(yolo_layer.get_output(0))
    network.mark_output(yolo_layer.get_output(1))

    return yolo_layer












# 假设 det0, det1, det2 是 IConvolutionLayer 的对象（或直接是 ITensor）
# 如果是 IConvolutionLayer，则传 det->get_output(0)，如果已经是 ITensor 则直接用
det_tensors = [det0.get_output(0), det1.get_output(0), det2.get_output(0)]
yolo_layer = add_yolo_layer_py(network, weight_map, "model.24", det_tensors=det_tensors, is_segmentation=False)



或者：
# 你之前查看到 concat 层 index 为 293
yolo_layer = add_yolo_layer_py(network, weight_map, "model.24", concat_layer_index=293, is_segmentation=False)

