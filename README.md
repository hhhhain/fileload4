import os
import pickle
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
import time
import math


# === GPU / TRT ===
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化 CUDA 上下文

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

tensorrt_version = trt.__version__
major_version = int(tensorrt_version.split('.')[0])
minor_version = int(tensorrt_version.split('.')[1])
device = torch.cuda.current_device()
total_memory = torch.cuda.get_device_properties(device).total_memory

class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }
    WARMUP = 10
    REPEAT = 10
    TEST = True # 部署时将其设置为False

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=self.device)
        self.__init_engine()

    def print_bindings(self, engine):
        nb = engine.num_bindings
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            io = "Input " if is_input else "Output"
            print(f"[Binding {i}] {io} name='{name}' dtype={dtype} shape={shape}")

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        # with trt.Runtime(logger) as runtime:
        #     model = runtime.deserialize_cuda_engine(self.weight.read_bytes())
        with open(self.weight, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(logger)
        model = runtime.deserialize_cuda_engine(engine_bytes)
        self.print_bindings(model)

        context = model.create_execution_context()

        num_bindings = model.num_bindings # 6
        names = [model.get_binding_name(i) for i in range(num_bindings)] # 6->['images','kpt','onnx::Shape_800','onnx::Reshape_819','onnx::Reshape_838','output0']
        num_inputs = sum([1 for i in range(num_bindings) if model.binding_is_input(i)]) # 1
        num_outputs = num_bindings - num_inputs # 5

        self.bindings: List[int] = [0] * (num_inputs + num_outputs)  # TensorRT 8
        self.num_bindings = num_bindings
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs] # ['images']
        self.output_names = names[num_inputs:] # ['kpt','onnx::Shape_800','onnx::Reshape_819','onnx::Reshape_838','output0']
        self.idx = list(range(self.num_outputs))
        
        self.input_shape = model.get_binding_shape(0) # [-1,3,640,1088]
        self.H = self.input_shape[-2]
        self.W = self.input_shape[-1]

    def get_io_indices(self, engine):  # inputs-->[0], outputs[1,2,3,4,5]
        inputs, outputs = [], []
        for i in range(engine.num_bindings):
            (inputs if engine.binding_is_input(i) else outputs).append(i)
        assert len(inputs) == 1, f"期望 1 个输入，但找到 {len(inputs)} 个。"
        # yolov8-pose 通常 1 个输出，但也可能更多，这里支持多输出
        return inputs[0], outputs

    def allocate_buffers(self, context, engine, batch, H, W):
        # 设定动态形状
        inp_idx, out_indices = self.get_io_indices(engine)
        # 多 profile 时可设置 context.active_optimization_profile = k
        context.set_binding_shape(inp_idx, (batch, 3, H, W))

        # 查询真实 IO 形状
        binding_shapes = {}  # 长度为6
        binding_dptrs = {}  # 长度为6
        host_buffers = {}  # 长度为6

        for i in range(engine.num_bindings):
            shape = context.get_binding_shape(i)
            shape_tuple = tuple(shape)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            try:
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            except:
                print("shape is not tuple")
                nbytes = int(np.prod(shape_tuple)) * np.dtype(dtype).itemsize
            if engine.binding_is_input(i):
                # host 输入
                host_buffers[i] = np.empty(shape, dtype=dtype)
                dptr = cuda.mem_alloc(nbytes)
                binding_dptrs[i] = dptr
            else:
                # host 输出
                host_buffers[i] = np.empty(shape, dtype=dtype)
                dptr = cuda.mem_alloc(nbytes)
                binding_dptrs[i] = dptr
            binding_shapes[i] = shape

        # bindings 列表必须按 binding 索引顺序填充指针
        bindings = [int(binding_dptrs[i]) for i in range(engine.num_bindings)]
        return host_buffers, binding_dptrs, bindings, binding_shapes

    def forward(self, inputs): # numpy, Tensor
        print(f"input shape:{inputs.shape}")
        B, C, H, W = inputs.shape # [10,3,640,1088]
        # 分配显存
        host_bufs, dev_ptrs, bindings, bind_shapes = self.allocate_buffers(self.context, self.model, B, H, W)
        inp_idx, out_indices = self.get_io_indices(self.model)
        out_shapes = [bind_shapes[i] for i in out_indices]
        self.stream = cuda.Stream()
        start_evt = cuda.Event()
        end_evt = cuda.Event()

        host_bufs[inp_idx][...] = inputs
        if self.TEST:
            # warm up
            for _ in range(self.WARMUP):
                cuda.memcpy_htod_async(dev_ptrs[inp_idx], host_bufs[inp_idx], self.stream)
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                # 将所有输出拷回（便于后续 postprocess 维度推断）
                for oi in out_indices:
                    cuda.memcpy_dtoh_async(host_bufs[oi], dev_ptrs[oi], self.stream)
                self.stream.synchronize()
    
            # infer
            infer_times_ms = []
            post_times_ms = []
            for _ in range(self.REPEAT):
                # 推理（GPU计时：仅 enqueue->kernel->完成）
                start_evt.record(self.stream)
                cuda.memcpy_htod_async(dev_ptrs[inp_idx], host_bufs[inp_idx], self.stream)
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                for oi in out_indices:
                    cuda.memcpy_dtoh_async(host_bufs[oi], dev_ptrs[oi], self.stream)
                end_evt.record(self.stream)
                self.stream.synchronize()
                gpu_time = start_evt.time_till(end_evt)  # 毫秒
                infer_times_ms.append(gpu_time)
            return host_bufs[oi], infer_times_ms
        else:
            start_evt.record(self.stream)
            cuda.memcpy_htod_async(dev_ptrs[inp_idx], host_bufs[inp_idx], self.stream)
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            for oi in out_indices:
                cuda.memcpy_dtoh_async(host_bufs[oi], dev_ptrs[oi], self.stream)
            end_evt.record(self.stream)
            self.stream.synchronize()
            gpu_time = start_evt.time_till(end_evt)  # 毫秒
            return host_bufs[oi], gpu_time



if __name__ == "__main__":
    ENGINE_PATH = "weights/yolov8s-pose-h640w1088-fp16.trt"
    # ENGINE_PATH = "weights/yolov8s-pose-prune-sp0.3-h640w1088-fp16.trt"
    # ENGINE_PATH = "weights/yolov8s-pose-prune-sp0.5-h640w1088-fp16.trt"
    # qat
    # ENGINE_PATH = "weights/yolov8s-pose-qat-fix-h640w1088.trt"
    # ENGINE_PATH = "weights/yolov8s-pose-prune-sp0.3-qat-fix-h640w1088.trt"
    # ENGINE_PATH = "weights/yolov8s-pose-prune-sp0.5-qat-fix-h640w1088.trt"
    
    engine = TRTModule(ENGINE_PATH, device)
    for batch in range(1,17):
        inputs = torch.rand(batch, 3, 640, 1088)
        output, infer_time = engine(inputs)
        # print(output)
        # print(f"output shape:{output.shape}")
        print(f"infer time lists:{infer_time}")
        print(f"run {batch} imgs infer avg time:{sum(infer_time)/len(infer_time)}")







    
