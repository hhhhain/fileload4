        host_inputs = [batch_input_image.ravel()]  # [10. 640, 1088]-->ravel 10*640*1088
        host_inputs[0] = host_inputs[0].astype(self.input_dtype)
