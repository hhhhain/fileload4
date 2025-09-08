    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        ctypes.CDLL("/home/ma-user/work/myplugin/build/libpost_process_before_nms.so", mode=ctypes.RTLD_GLOBAL)
        # 初始化Plugin插件, 也可以提供一个logger替换None,记录信息
        trt.init_libnvinfer_plugins(None, "")
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.max_batch_size = 1
        context.set_binding_shape(0, (10, 3, 640, 1088))
        for binding in engine:
            log.info('bingding:{},{}'.format(binding, engine.get_binding_shape(binding)))

            # [+] 3. 使用新的逻辑计算 size
            shape = engine.get_binding_shape(binding)
            print(shape)
            shape_tuple = tuple(shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):            
                self.input_dtype = dtype
            else:   
                self.output_dtype = dtype
            # dtype = np.float32
            print('dtype is', dtype)
            # exit()
            # 检查是否存在动态维度 (-1)
            if shape[0] == -1:
                # 计算单个批次项的大小 (忽略第一个维度), 然后乘以我们定义的最大批量
                # size = trt.volume(shape[1:]) * self.max_batch_size
                size = (10,)+tuple(shape[1:])
            else:
                # 如果没有动态维度，则直接计算
                # size = trt.volume(shape) * self.max_batch_size
                size = shape
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            print('3333333333333333')
            host_mem = np.empty(size, dtype)  # 现在 size 会是一个正数
            print('host_mem is', host_mem.shape)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # print('cuda_mem is', host_mem.nbytes)
            # host_mem = np.empty(size, dtype)
            # cuda_mem = cuda.mem_alloc(nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                # print('add input done')
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                # print('add output done')
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                # exit()
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
        # for i in range(engine.num_bindings):
        #     name_for_trans = engine.get_binding_name(i)
        #     context.set_tensor_address(name_for_trans, bindings[i])
        
        inputs = torch.rand(10, 3, 640, 1088)
        for i in range(50):
            print('777777777777777')
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            print('4444444444444444444')
            # exit()
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            print('5555555555555')
            cuda.memcpy_dtoh_async(host_outputs[-1], cuda_outputs[-1], stream)
            print('6666666666666666666')


context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)就是执行到这一句的时候报错的。            
