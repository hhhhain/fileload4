        for i in range(50):
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
