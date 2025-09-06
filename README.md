            shape = engine.get_binding_shape(binding)
            print(shape)
            shape_tuple = tuple(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            self.dtype = dtype
            # dtype = np.float32
            # print('dtype is', dtype)
            # exit()
            # 检查是否存在动态维度 (-1)
            if shape[0] == -1:
                # 计算单个批次项的大小 (忽略第一个维度), 然后乘以我们定义的最大批量
                # size = trt.volume(shape[1:]) * self.max_batch_size
                size = (10,)+tuple(shape[1:])
            else:
                # 如果没有动态维度，则直接计算
                # size = trt.volume(shape) * self.max_batch_size
                size = shape
