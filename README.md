        context.set_binding_shape(0, (10, 3, 640, 1088))
        for binding in engine:
            log.info('bingding:{},{}'.format(binding, engine.get_binding_shape(binding)))

            # [+] 3. 使用新的逻辑计算 size
            shape = engine.get_binding_shape(binding)
            print(shape)
            shape_tuple = tuple(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # 检查是否存在动态维度 (-1)
            if shape[0] == -1:
                # 计算单个批次项的大小 (忽略第一个维度), 然后乘以我们定义的最大批量
                # size = trt.volume(shape[1:]) * self.max_batch_size
                size = (10,)+tuple(shape[1:])
            else:
                # 如果没有动态维度，则直接计算
                # size = trt.volume(shape) * self.max_batch_size
                size = shape
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = np.empty(size, dtype)  # 现在 size 会是一个正数
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # host_mem = np.empty(size, dtype)
            # cuda_mem = cuda.mem_alloc(nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
