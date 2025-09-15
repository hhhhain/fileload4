    def infer(self, batch_input_image, task):
        cuda.memset_d32(self.out_mem_addr, 0, self.out_size)
        log.info(f"batch image shape:{batch_input_image.shape}")
        threading.Thread.__init__(self)
        self.ctx.push()
        stream = self.stream
        context = self.context
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        host_inputs = [batch_input_image.ravel()]  # [10. 640, 1088]-->ravel 10*640*1088
        start = time.time()
        start_evt = cuda.Event()
        end_evt = cuda.Event()
        start_evt.record(stream) 
        print('host_inputs[0].shape is', host_inputs[0][115000:115020])
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        end_evt.record(stream) 
        end_evt.synchronize() 
        print(f"HTOD memcpy: {start_evt.time_till(end_evt):.2f} ms")
        start_evt.record(stream) 
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        end_evt.record(stream) 
        end_evt.synchronize() 
        print(f"infer: {start_evt.time_till(end_evt):.2f} ms")        
        start_evt.record(stream) 
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        
        end_evt.record(stream) 
        end_evt.synchronize() 
        print(f"DTOH memcpy: {start_evt.time_till(end_evt):.2f} ms")        
        stream.synchronize()
        end = time.time()
        self.ctx.pop()
        output = host_outputs[0]
        output_format = output
        log.info(f"output shape: {host_outputs[0].shape}")
        log.info(f"output value: {host_outputs[0][:40]}")
        log.info("{} inference Speed: {:.2f}ms".format(task, (end - start) * 1000))

        return output
