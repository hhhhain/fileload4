    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
pycuda._driver.LogicError: cuMemcpyHtoDAsync failed: invalid argument
