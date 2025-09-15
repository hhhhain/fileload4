# 保存
np.save("input_tensor.npy", host_inputs[0])

# 加载
loaded = np.load("input_tensor.npy")
print(loaded.shape, loaded.dtype)
