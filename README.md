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
