        for i in range(50):
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_outputs[-1], cuda_outputs[-1], stream)

            这么warmup正确吗？我发现不热身传回来的rescount是一万多，热身传回来的是0
