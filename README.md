    det1 = network.get_layer(242).get_output(0)
    det2 = network.get_layer(267).get_output(0)
    det3 = network.get_layer(292).get_output(0)
    add_yolo_layer_py(network, det_tensors=[det1,det2,det3], concat_layer_index=293, is_segmentation=False)


    def add_yolo_layer_py(network, det_tensors=None, concat_layer_index=None, is_segmentation=False):
    """
    - network: trt.INetworkDefinition
    - weight_map: 你的权重字典（用于 getAnchors/get strides）
    - lname: e.g. "model.24"
    - det_tensors: 列表 of ITensor (三个 detection head 的输出), 优先使用这个
    - concat_layer_index: 如果你想直接处理 concat 层，传该 layer index（int）
    - 返回: 插入的 plugin layer (ILayer)
    """
    kMaxNumOutputBbox = 1000
    kNumClass = 26
    kInputW = 640
    kInputH = 1088
    is_segmentation = 0

    # 1) 确保插件库被加载并初始化（在外部只需做一次也可以）
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

    # 4) 获取scales
    scales = np.array([8, 16, 32], dtype=np.float32)

    # 5) 构造 kernels 内容 —— 必须与 C++ 插件期望的内存布局一致
    # 在 C++ 里 YoloKernel 结构体可能像: struct YoloKernel { int width; int height; float anchors[6]; }
    # 因此在 Python 需要按相同字节序打包 (int32, int32, float32*x)
    anchors = [
        [10,13, 16,30, 33,23],       # s8
        [30,61, 62,45, 59,119],      # s16
        [116,90, 156,198, 373,326]   # s32
    ]
    kernels = []

    for i in range(len(anchors)):
        w = kInputW / scales[i]
        h = kInputH / scales[i]
        # 一个 YoloKernel = (width, height, anchors)
        kernel = [w, h] + anchors[i]
        kernels.extend(kernel)
    kernels = np.array(kernels, dtype=np.float32)    
    kernels_field = trt.PluginField("kernels", kernels, trt.PluginFieldType.FLOAT32) 

    # 6) 组装 PluginFieldCollection
    fields = [f_netinfo, kernels_field]
    field_collection = trt.PluginFieldCollection(fields)

    # 7) 创建 plugin 实例
    plugin_obj = creator.create_plugin(name="yololayer", field_collection=field_collection)
    if plugin_obj is None:
        raise RuntimeError("creator.create_plugin returned None")

    # 8) 选择输入 tensors：优先使用 det_tensors，否则用 concat_layer_index
    inputs = det_tensors
    concat_tensor = network.get_layer(concat_layer_index).get_output(0)
    # 如果 concat_tensor 在 network outputs 中已经被标记 output，需要先 unmark
    try:
        network.unmark_output(concat_tensor)
    except Exception:
        # 如果没被标记，这会抛错或返回；忽略
        pass

    # 9) 把 plugin 插入网络
    yolo_layer = network.add_plugin_v2(inputs=inputs, plugin=plugin_obj)
    if yolo_layer is None:
        raise RuntimeError("network.add_plugin_v2 returned None")

    # 10) 命名并标记输出（示例）
    yolo_layer.get_output(0).name = "yolo_out_post"
    network.mark_output(yolo_layer.get_output(0))

    return yolo_layer
