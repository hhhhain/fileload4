import numpy as np

# 错误解读的 float32 数据
wrong_fp32 = np.array([7.523438, 5.960938, 17.765625, 14.757812, 0.000002], dtype=np.float32)

# 把它当作原始字节流
raw_bytes = wrong_fp32.tobytes()

# 用 float16 方式重解读
fp16_view = np.frombuffer(raw_bytes, dtype=np.float16)

# 再转回 float32
corrected_fp32 = fp16_view.astype(np.float32)

print("FP16 view:", fp16_view)
print("Corrected FP32:", corrected_fp32)
