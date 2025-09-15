            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            if engine.binding_is_input(binding): 
                # dtype = np.float32           
                self.input_dtype = dtype
            else:   
                self.output_dtype = dtype            
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            print('host_mem is', len(host_mem))
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
