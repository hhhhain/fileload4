        start_evt = cuda.Event()
        end_evt = cuda.Event()

        start_evt.record(stream) 
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        end_evt.record(stream) 
        end_evt.synchronize() 
        print(f"HTOD memcpy: {start_evt.time_till(end_evt):.2f} ms")
        
        # Run inference.
        # context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        start_evt.record(stream) 
        # context.profiler = trt.Profiler()
        # context.execute_async_v3(bindings=bindings, stream_handle=stream.handle)
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        end_evt.record(stream) 
        end_evt.synchronize() 
        print(f"infer: {start_evt.time_till(end_evt):.2f} ms")        
        # Transfer predictions back from the GPU.
        # for 
        log.info(f"length host_outputs:{len(host_outputs)}")
        # cuda.memcpy_dtoh_async(host_outputs[-1], cuda_outputs[-1], stream)
        # # for i in range(len(cuda_outputs)):
        # #     cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # #     # host_outputs[i] = host_outputs[i].reshape(self.output_shape)
        # # Synchronize the stream
        start_evt.record(stream) 
        cuda.memcpy_dtoh_async(host_outputs[-1], cuda_outputs[-1], stream)
        end_evt.record(stream) 
        end_evt.synchronize() 
        print(f"DTOH memcpy: {start_evt.time_till(end_evt):.2f} ms") 
