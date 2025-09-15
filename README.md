  //load strides from Detect layer
  assert(weightMap.find(lname + ".strides") != weightMap.end() && "Not found `strides`, please check gen_wts.py!!!");
  Weights strides = weightMap[lname + ".strides"];
  auto *p = (const float*)(strides.values);
  std::vector<int> scales(p, p + strides.count);

  std::vector<YoloKernel> kernels;
  for (size_t i = 0; i < anchors.size(); i++) {
    YoloKernel kernel;
    kernel.width = kInputW / scales[i];
    kernel.height = kInputH / scales[i];
    memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
    // push_back用来在vector末尾添加一个元素.
    kernels.push_back(kernel);
  }
  plugin_fields[1].data = &kernels[0];
  plugin_fields[1].length = kernels.size();
  plugin_fields[1].name = "kernels";
  plugin_fields[1].type = PluginFieldType::kFLOAT32;
  我根据这个c++我自己翻译成了Python，但是没解析出head里面的w h这些值，我翻译是不是错了：
    scales = [8, 16, 32]
    kernels_bytes = b""
    anchors = [
        [10,13, 16,30, 33,23],       # s8
        [30,61, 62,45, 59,119],      # s16
        [116,90, 156,198, 373,326]   # s32
    ]
    kernels = []
    for i in range(len(anchors)):
        w = int(kInputW / scales[i])   # int
        h = int(kInputH / scales[i])   # int
        kernels_bytes += struct.pack("ii", w, h)
        kernels_bytes += struct.pack("6f", *anchors[i])
    kernels = np.frombuffer(kernels_bytes, dtype=np.uint8)
    kernels_field = trt.PluginField("kernels", kernels, trt.PluginFieldType.UNKNOWN)
    fields = [f_netinfo, kernels_field]
    field_collection = trt.PluginFieldCollection(fields)

    
