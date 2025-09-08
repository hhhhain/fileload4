    kMaxNumOutputBbox = 1
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
    netinfo = np.array([kNumClass, kInputW, kInputH, kMaxNumOutputBbox, int(is_segmentation)], dtype=np.float32)
    f_netinfo = trt.PluginField("netinfo", netinfo, trt.PluginFieldType.FLOAT32)

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


然后打印
IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT {
  // 下面的断言是做限制，第一个字段名必须是netinfo，第二个必须是kernels
  assert(fc->nbFields == 2);
  assert(strcmp(fc->fields[0].name, "netinfo") == 0);
  assert(strcmp(fc->fields[1].name, "kernels") == 0);

  // 接下来具体解析第一个字段netinfo，这是一个int数组。
  // 分别是类别数、输入宽高、每张图片最多输出多少个、是否是分割模型
  int *p_netinfo = (int*)(fc->fields[0].data);
  int class_count = p_netinfo[0];
  int input_w = p_netinfo[1];
  int input_h = p_netinfo[2];
  int max_output_object_count = p_netinfo[3];
  bool is_segmentation = (bool)p_netinfo[4];


  std::cout << "max_output_object_count=" << max_output_object_count
          << std::endl;

你帮我看问题出在哪，得到max_output_object_count=1065353216
