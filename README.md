# 1. torch tensor -> numpy
host_inputs = [inputs.cpu().numpy().ravel()]

# 2. 转成和 engine 输入绑定一致的 dtype
host_inputs[0] = host_inputs[0].astype(trt.nptype(self.input_dtype))  
