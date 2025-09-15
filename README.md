    arr = np.array([w, h] + anchors[i], dtype=np.float32)  # 注意：这里 int 也用 float32, C++端可能能接受
    kernels_list.append(arr)

# 转成连续内存
kernels_array = np.concatenate(kernels_list).astype(np.float32)

kernels_field = trt.PluginField("kernels", kernels_array, trt.PluginFieldType.FLOAT32)
