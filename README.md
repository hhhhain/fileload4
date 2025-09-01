class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

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
            f"output shape:{self.output_shape}")  # [-1,N,DET_NUM + SEG_NUM + POSE_NUM] output shape:(-1, 56, 14280) (90001, 1, 1)

        self.max_batch_size = 1
        context.set_binding_shape(0, (10, 3, 640, 1088))
        for binding in engine:
            log.info('bingding:{},{}'.format(binding, engine.get_binding_shape(binding)))

            # [+] 3. 使用新的逻辑计算 size
            shape = engine.get_binding_shape(binding)
            print(shape)
            shape_tuple = tuple(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # 检查是否存在动态维度 (-1)
            if shape[0] == -1:
                # 计算单个批次项的大小 (忽略第一个维度), 然后乘以我们定义的最大批量
                # size = trt.volume(shape[1:]) * self.max_batch_size
                size = shape[1:]
            else:
                # 如果没有动态维度，则直接计算
                # size = trt.volume(shape) * self.max_batch_size
                size = shape
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = np.empty(size, dtype)  # 现在 size 会是一个正数
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # host_mem = np.empty(size, dtype)
            # cuda_mem = cuda.mem_alloc(nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = 10
        # self.det_output_size = host_outputs[0].shape[0]
        self.det_output_size = host_outputs[-1].shape[0]
        for i in range(engine.num_bindings):
            name_for_trans = engine.get_binding_name(i)
            context.set_tensor_address(name_for_trans, bindings[i])
