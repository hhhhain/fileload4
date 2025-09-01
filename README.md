 # set input tensor address
        for name, buffer in zip(self.input_tensor_names, buffers_device):
            self.context.set_tensor_address(name, int(buffer))
        # set output tensor allocator
        for name in self.output_tensor_names:
            self.context.set_tensor_address(name, 0) # set nullptr
            self.context.set_output_allocator(name, self.output_allocator)
        # The do_inference function will return a list of outputs
