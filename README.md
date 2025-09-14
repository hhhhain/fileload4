        inputs = torch.rand(10, 3, 640, 1088)



        host_inputs = [inputs.ravel()]  # [10. 640, 1088]-->ravel 10*640*1088
        # host_bufs[0][...] = batch_input_image
        # print('host_inputs', host_inputs)
        # 把输入的类型转换为trt绑定的类型.
        host_inputs[0] = host_inputs[0].dtype(self.input_dtype)


        for i in range(50):
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
