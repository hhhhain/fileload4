            print('size is ', size)     
            print('dtype is ', dtype)     
            host_mem = np.empty(size, dtype)  # 现在 size 会是一个正数
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            # print('host_mem.nbytes is', host_mem.nbytes)
            # print('cuda_mem is', host_mem.nbytes)
            # host_mem = np.empty(size, dtype)
            # cuda_mem = cuda.mem_alloc(nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                print('add input done')
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                print('add output done')
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
