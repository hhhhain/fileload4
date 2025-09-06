    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        # # add dynamic surpport
        for i in range(engine.num_bindings):
            log.info(f"{i}, {engine.get_binding_name(i)}, {engine.get_binding_shape(i)}")
        input_binding_idx = engine.get_binding_index("input") if engine.get_binding_index(
            "input") != -1 else engine.get_binding_index("images")
        input_shape = engine.get_binding_shape(input_binding_idx)
        output_binding_idx = engine.get_binding_index("output") if engine.get_binding_index(
            "output") != -1 else engine.get_binding_index("output0")
        self.output_shape = engine.get_binding_shape(output_binding_idx)
        log.info(f"input shape:{input_shape}")  # [-1,3,640,640]
        log.info(
            f"output shape:{self.output_shape}") 


2025-09-06 15:06:50 INFO object_detect.__init__(object_detect.py:105): 0, images, (10, 3, 640, 1088)
2025-09-06 15:06:50 INFO object_detect.__init__(object_detect.py:105): 1, output0, (10, 42840, 31)
[09/06/2025-15:06:50] [TRT] [E] 3: Cannot find binding of given name: input
[09/06/2025-15:06:50] [TRT] [E] 3: Cannot find binding of given name: output
2025-09-06 15:06:50 INFO object_detect.__init__(object_detect.py:112): input shape:(10, 3, 640, 1088)
2025-09-06 15:06:50 INFO object_detect.__init__(object_detect.py:113): output shape:(10, 42840, 31)
2025-09-06 15:06:50 INFO object_detect.__init__(object_detect.py:169): bingding:images,(10, 3, 640, 1088)
(10, 3, 640, 1088)
2025-09-06 15:06:50 INFO object_detect.__init__(object_detect.py:169): bingding:output0,(10, 42840, 31)
            
