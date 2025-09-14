        host_inputs = [batch_input_image.ravel()]  
        host_inputs[0] = host_inputs[0].astype(self.input_dtype)
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        我这个代码是对的，但我不想用imgae，你能帮我修改成zeros吗？
