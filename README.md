
        初始化热身：
        for i in range(50):
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_outputs[-1], cuda_outputs[-1], stream)
        stream.synchronize()
        # exit()
        

正式推理
    def infer(self, batch_input_image, task):
        log.info(f"batch image shape:{batch_input_image.shape}")
        threading.Thread.__init__(self)
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        host_inputs = [batch_input_image.ravel()]  # [10. 640, 1088]-->ravel 10*640*1088
        host_inputs[0] = host_inputs[0].astype(self.input_dtype)
        start_evt = cuda.Event()
        end_evt = cuda.Event()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        self.ctx.pop()
        output = host_outputs[0]
        output_format = output
        log.info(f"output shape: {host_outputs}")
        log.info(f"output shape: {host_outputs[0].shape}")
        log.info(f"output shape: {host_outputs[0][:80]}")
我不理解，如果我热身了，最后host_outputs全是0.如果我不热身，host_outputs就是正确的结果。为什么
