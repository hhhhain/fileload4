ma-user@notebook-2f5fce31-4e34-4250-b6e1-6d89ab1980d6:~/work/copy/files/video-deal-search/video-deal-service$ trtexec --loadEngine=/home/ma-user/infer/video-deal-search/video-deal-service/pt/CP26classes_epoch_180.trt --verbose
&&&& RUNNING TensorRT.trtexec [TensorRT v8601] # trtexec --loadEngine=/home/ma-user/infer/video-deal-search/video-deal-service/pt/CP26classes_epoch_180.trt --verbose
[08/31/2025-21:07:07] [I] === Model Options ===
[08/31/2025-21:07:07] [I] Format: *
[08/31/2025-21:07:07] [I] Model: 
[08/31/2025-21:07:07] [I] Output:
[08/31/2025-21:07:07] [I] === Build Options ===
[08/31/2025-21:07:07] [I] Max batch: 1
[08/31/2025-21:07:07] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default
[08/31/2025-21:07:07] [I] minTiming: 1
[08/31/2025-21:07:07] [I] avgTiming: 8
[08/31/2025-21:07:07] [I] Precision: FP32
[08/31/2025-21:07:07] [I] LayerPrecisions: 
[08/31/2025-21:07:07] [I] Layer Device Types: 
[08/31/2025-21:07:07] [I] Calibration: 
[08/31/2025-21:07:07] [I] Refit: Disabled
[08/31/2025-21:07:07] [I] Version Compatible: Disabled
[08/31/2025-21:07:07] [I] TensorRT runtime: full
[08/31/2025-21:07:07] [I] Lean DLL Path: 
[08/31/2025-21:07:07] [I] Tempfile Controls: { in_memory: allow, temporary: allow }
[08/31/2025-21:07:07] [I] Exclude Lean Runtime: Disabled
[08/31/2025-21:07:07] [I] Sparsity: Disabled
[08/31/2025-21:07:07] [I] Safe mode: Disabled
[08/31/2025-21:07:07] [I] Build DLA standalone loadable: Disabled
[08/31/2025-21:07:07] [I] Allow GPU fallback for DLA: Disabled
[08/31/2025-21:07:07] [I] DirectIO mode: Disabled
[08/31/2025-21:07:07] [I] Restricted mode: Disabled
[08/31/2025-21:07:07] [I] Skip inference: Disabled
[08/31/2025-21:07:07] [I] Save engine: 
[08/31/2025-21:07:07] [I] Load engine: /home/ma-user/infer/video-deal-search/video-deal-service/pt/CP26classes_epoch_180.trt
[08/31/2025-21:07:07] [I] Profiling verbosity: 0
[08/31/2025-21:07:07] [I] Tactic sources: Using default tactic sources
[08/31/2025-21:07:07] [I] timingCacheMode: local
[08/31/2025-21:07:07] [I] timingCacheFile: 
[08/31/2025-21:07:07] [I] Heuristic: Disabled
[08/31/2025-21:07:07] [I] Preview Features: Use default preview flags.
[08/31/2025-21:07:07] [I] MaxAuxStreams: -1
[08/31/2025-21:07:07] [I] BuilderOptimizationLevel: -1
[08/31/2025-21:07:07] [I] Input(s)s format: fp32:CHW
[08/31/2025-21:07:07] [I] Output(s)s format: fp32:CHW
[08/31/2025-21:07:07] [I] Input build shapes: model
[08/31/2025-21:07:07] [I] Input calibration shapes: model
[08/31/2025-21:07:07] [I] === System Options ===
[08/31/2025-21:07:07] [I] Device: 0
[08/31/2025-21:07:07] [I] DLACore: 
[08/31/2025-21:07:07] [I] Plugins:
[08/31/2025-21:07:07] [I] setPluginsToSerialize:
[08/31/2025-21:07:07] [I] dynamicPlugins:
[08/31/2025-21:07:07] [I] ignoreParsedPluginLibs: 0
[08/31/2025-21:07:07] [I] 
[08/31/2025-21:07:07] [I] === Inference Options ===
[08/31/2025-21:07:07] [I] Batch: 1
[08/31/2025-21:07:07] [I] Input inference shapes: model
[08/31/2025-21:07:07] [I] Iterations: 10
[08/31/2025-21:07:07] [I] Duration: 3s (+ 200ms warm up)
[08/31/2025-21:07:07] [I] Sleep time: 0ms
[08/31/2025-21:07:07] [I] Idle time: 0ms
[08/31/2025-21:07:07] [I] Inference Streams: 1
[08/31/2025-21:07:07] [I] ExposeDMA: Disabled
[08/31/2025-21:07:07] [I] Data transfers: Enabled
[08/31/2025-21:07:07] [I] Spin-wait: Disabled
[08/31/2025-21:07:07] [I] Multithreading: Disabled
[08/31/2025-21:07:07] [I] CUDA Graph: Disabled
[08/31/2025-21:07:07] [I] Separate profiling: Disabled
[08/31/2025-21:07:07] [I] Time Deserialize: Disabled
[08/31/2025-21:07:07] [I] Time Refit: Disabled
[08/31/2025-21:07:07] [I] NVTX verbosity: 0
[08/31/2025-21:07:07] [I] Persistent Cache Ratio: 0
[08/31/2025-21:07:07] [I] Inputs:
[08/31/2025-21:07:07] [I] === Reporting Options ===
[08/31/2025-21:07:07] [I] Verbose: Enabled
[08/31/2025-21:07:07] [I] Averages: 10 inferences
[08/31/2025-21:07:07] [I] Percentiles: 90,95,99
[08/31/2025-21:07:07] [I] Dump refittable layers:Disabled
[08/31/2025-21:07:07] [I] Dump output: Disabled
[08/31/2025-21:07:07] [I] Profile: Disabled
[08/31/2025-21:07:07] [I] Export timing to JSON file: 
[08/31/2025-21:07:07] [I] Export output to JSON file: 
[08/31/2025-21:07:07] [I] Export profile to JSON file: 
[08/31/2025-21:07:07] [I] 
[08/31/2025-21:07:08] [I] === Device Information ===
[08/31/2025-21:07:08] [I] Selected Device: Tesla T4
[08/31/2025-21:07:08] [I] Compute Capability: 7.5
[08/31/2025-21:07:08] [I] SMs: 40
[08/31/2025-21:07:08] [I] Device Global Memory: 15102 MiB
[08/31/2025-21:07:08] [I] Shared Memory per SM: 64 KiB
[08/31/2025-21:07:08] [I] Memory Bus Width: 256 bits (ECC enabled)
[08/31/2025-21:07:08] [I] Application Compute Clock Rate: 1.59 GHz
[08/31/2025-21:07:08] [I] Application Memory Clock Rate: 5.001 GHz
[08/31/2025-21:07:08] [I] 
[08/31/2025-21:07:08] [I] Note: The application clock rates do not reflect the actual clock rates that the GPU is currently running at.
[08/31/2025-21:07:08] [I] 
[08/31/2025-21:07:08] [I] TensorRT version: 8.6.1
[08/31/2025-21:07:08] [I] Loading standard plugins
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::BatchedNMSDynamic_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::BatchedNMS_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::BatchTilePlugin_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::Clip_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::CoordConvAC version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::CropAndResizeDynamic version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::CropAndResize version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::DecodeBbox3DPlugin version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::DetectionLayer_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::EfficientNMS_Explicit_TF_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::EfficientNMS_Implicit_TF_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::EfficientNMS_ONNX_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::EfficientNMS_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::FlattenConcat_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::GenerateDetection_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::GridAnchor_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::GridAnchorRect_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::InstanceNormalization_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::InstanceNormalization_TRT version 2
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::LReLU_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::ModulatedDeformConv2d version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::MultilevelCropAndResize_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::MultilevelProposeROI_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::MultiscaleDeformableAttnPlugin_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::NMSDynamic_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::NMS_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::Normalize_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::PillarScatterPlugin version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::PriorBox_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::ProposalDynamic version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::ProposalLayer_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::Proposal version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::PyramidROIAlign_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::Region_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::Reorg_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::ResizeNearest_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::ROIAlign_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::RPROI_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::ScatterND version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::SpecialSlice_TRT version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::Split version 1
[08/31/2025-21:07:08] [V] [TRT] Registered plugin creator - ::VoxelGeneratorPlugin version 1
[08/31/2025-21:07:08] [I] Engine loaded in 0.0137625 sec.
[08/31/2025-21:07:08] [I] [TRT] Loaded engine size: 15 MiB
[08/31/2025-21:07:08] [V] [TRT] Deserialization required 28040 microseconds.
[08/31/2025-21:07:08] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +15, now: CPU 0, GPU 15 (MiB)
[08/31/2025-21:07:08] [I] Engine deserialized in 0.0357642 sec.
[08/31/2025-21:07:08] [V] [TRT] Total per-runner device persistent memory is 1570304
[08/31/2025-21:07:08] [V] [TRT] Total per-runner host persistent memory is 279696
[08/31/2025-21:07:08] [V] [TRT] Allocated activation device memory of size 289871872
[08/31/2025-21:07:08] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +278, now: CPU 0, GPU 293 (MiB)
[08/31/2025-21:07:08] [V] [TRT] CUDA lazy loading is enabled.
[08/31/2025-21:07:08] [I] Setting persistentCacheLimit to 0 bytes.
[08/31/2025-21:07:08] [V] Using enqueueV3.
[08/31/2025-21:07:08] [I] Using random values for input images
[08/31/2025-21:07:08] [I] Input binding for images with dimensions 10x3x640x1088 is created.
[08/31/2025-21:07:08] [I] Output binding for output0 with dimensions 10x42840x31 is created.
[08/31/2025-21:07:08] [I] Starting inference
[08/31/2025-21:07:11] [I] Warmup completed 5 queries over 200 ms
[08/31/2025-21:07:11] [I] Timing trace has 136 queries over 3.09059 s
[08/31/2025-21:07:11] [I] 
[08/31/2025-21:07:11] [I] === Trace details ===
[08/31/2025-21:07:11] [I] Trace averages of 10 runs:
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 22.0757 ms - Host latency: 27.4924 ms (enqueue 0.801474 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 21.6743 ms - Host latency: 27.0907 ms (enqueue 0.825894 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 21.752 ms - Host latency: 27.1687 ms (enqueue 0.885034 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 22.8301 ms - Host latency: 28.2471 ms (enqueue 0.818127 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 23.9037 ms - Host latency: 29.3221 ms (enqueue 0.929504 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 21.8945 ms - Host latency: 27.3124 ms (enqueue 0.782336 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 21.7782 ms - Host latency: 27.1939 ms (enqueue 0.76748 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 22.0159 ms - Host latency: 27.4319 ms (enqueue 0.75437 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 24.1034 ms - Host latency: 29.5187 ms (enqueue 0.763428 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 22.412 ms - Host latency: 27.8275 ms (enqueue 0.751147 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 21.9789 ms - Host latency: 27.3929 ms (enqueue 0.739331 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 21.8812 ms - Host latency: 27.2964 ms (enqueue 0.789209 ms)
[08/31/2025-21:07:11] [I] Average on 10 runs - GPU latency: 23.3051 ms - Host latency: 28.7206 ms (enqueue 0.771973 ms)
[08/31/2025-21:07:11] [I] 
[08/31/2025-21:07:11] [I] === Performance summary ===
[08/31/2025-21:07:11] [I] Throughput: 44.0045 qps
[08/31/2025-21:07:11] [I] Latency: min = 26.8655 ms, max = 31.8876 ms, mean = 27.9016 ms, median = 27.4583 ms, percentile(90%) = 29.5156 ms, percentile(95%) = 29.9088 ms, percentile(99%) = 31.5781 ms
[08/31/2025-21:07:11] [I] Enqueue Time: min = 0.684082 ms, max = 1.16016 ms, mean = 0.79691 ms, median = 0.771881 ms, percentile(90%) = 0.862427 ms, percentile(95%) = 1.10266 ms, percentile(99%) = 1.15564 ms
[08/31/2025-21:07:11] [I] H2D Latency: min = 3.38843 ms, max = 3.40283 ms, mean = 3.39401 ms, median = 3.39343 ms, percentile(90%) = 3.39789 ms, percentile(95%) = 3.39929 ms, percentile(99%) = 3.40131 ms
[08/31/2025-21:07:11] [I] GPU Compute Time: min = 21.4507 ms, max = 26.4669 ms, mean = 22.4855 ms, median = 22.0431 ms, percentile(90%) = 24.1021 ms, percentile(95%) = 24.494 ms, percentile(99%) = 26.1577 ms
[08/31/2025-21:07:11] [I] D2H Latency: min = 2.02002 ms, max = 2.02661 ms, mean = 2.02203 ms, median = 2.02185 ms, percentile(90%) = 2.02319 ms, percentile(95%) = 2.02441 ms, percentile(99%) = 2.02539 ms
[08/31/2025-21:07:11] [I] Total Host Walltime: 3.09059 s
[08/31/2025-21:07:11] [I] Total GPU Compute Time: 3.05803 s
[08/31/2025-21:07:11] [W] * GPU compute time is unstable, with coefficient of variance = 4.75734%.
[08/31/2025-21:07:11] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[08/31/2025-21:07:11] [I] Explanations of the performance metrics are printed in the verbose logs.
[08/31/2025-21:07:11] [V] 
[08/31/2025-21:07:11] [V] === Explanations of the performance metrics ===
[08/31/2025-21:07:11] [V] Total Host Walltime: the host walltime from when the first query (after warmups) is enqueued to when the last query is completed.
[08/31/2025-21:07:11] [V] GPU Compute Time: the GPU latency to execute the kernels for a query.
[08/31/2025-21:07:11] [V] Total GPU Compute Time: the summation of the GPU Compute Time of all the queries. If this is significantly shorter than Total Host Walltime, the GPU may be under-utilized because of host-side overheads or data transfers.
[08/31/2025-21:07:11] [V] Throughput: the observed throughput computed by dividing the number of queries by the Total Host Walltime. If this is significantly lower than the reciprocal of GPU Compute Time, the GPU may be under-utilized because of host-side overheads or data transfers.
[08/31/2025-21:07:11] [V] Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.
[08/31/2025-21:07:11] [V] H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.
[08/31/2025-21:07:11] [V] D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.
[08/31/2025-21:07:11] [V] Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a single query.
[08/31/2025-21:07:11] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8601] # trtexec --loadEngine=/home/ma-user/infer/video-deal-search/video-deal-service/pt/CP26classes_epoch_180.trt --verbose
ma-user@notebook-2f5fce31-4e34-4250-b6e1-6d89ab1980d6:~/work/copy/files/video-deal-search/video-deal-s
