import numpy as np
import struct

anchors = [
    [10,13, 16,30, 33,23],       # s8
    [30,61, 62,45, 59,119],      # s16
    [116,90, 156,198, 373,326]   # s32
]

scales = [8, 16, 32]
kernels_bytes = b""

for i in range(len(anchors)):
    w = int(kInputW / scales[i])   # int
    h = int(kInputH / scales[i])   # int
    # 先 pack 两个 int32
    kernels_bytes += struct.pack("ii", w, h)
    # 再 pack 六个 float32
    kernels_bytes += struct.pack("6f", *anchors[i])

# 转成 numpy (uint8) 给 PluginField
kernels = np.frombuffer(kernels_bytes, dtype=np.uint8)
kernels_field = trt.PluginField("kernels", kernels, trt.PluginFieldType.UNKNOWN)
