    m._weight_quantizer.amax = np.maximum(amax, 0.001)  # 避免 0        
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/torch/_tensor.py", line 1030, in __array__
    return self.numpy()
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
