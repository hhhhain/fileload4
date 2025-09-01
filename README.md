&&&& RUNNING TensorRT.trtexec [TensorRT v8205] # trtexec --loadEngine=/home/ma-user/work/video-deal-service/weights/CP26classes_epoch_180_fp16_bs10_640_1088_fromexec.trt
[09/01/2025-18:06:59] [I] === Model Options ===
[09/01/2025-18:06:59] [I] Format: *
[09/01/2025-18:06:59] [I] Model: 
[09/01/2025-18:06:59] [I] Output:
[09/01/2025-18:06:59] [I] === Build Options ===
[09/01/2025-18:06:59] [I] Max batch: 1
[09/01/2025-18:06:59] [I] Workspace: 16 MiB
[09/01/2025-18:06:59] [I] minTiming: 1
[09/01/2025-18:06:59] [I] avgTiming: 8
[09/01/2025-18:06:59] [I] Precision: FP32
[09/01/2025-18:06:59] [I] Calibration: 
[09/01/2025-18:06:59] [I] Refit: Disabled
[09/01/2025-18:06:59] [I] Sparsity: Disabled
[09/01/2025-18:06:59] [I] Safe mode: Disabled
[09/01/2025-18:06:59] [I] DirectIO mode: Disabled
[09/01/2025-18:06:59] [I] Restricted mode: Disabled
[09/01/2025-18:06:59] [I] Save engine: 
[09/01/2025-18:06:59] [I] Load engine: /home/ma-user/work/video-deal-service/weights/CP26classes_epoch_180_fp16_bs10_640_1088_fromexec.trt
[09/01/2025-18:06:59] [I] Profiling verbosity: 0
[09/01/2025-18:06:59] [I] Tactic sources: Using default tactic sources
[09/01/2025-18:06:59] [I] timingCacheMode: local
[09/01/2025-18:06:59] [I] timingCacheFile: 
[09/01/2025-18:06:59] [I] Input(s)s format: fp32:CHW
[09/01/2025-18:06:59] [I] Output(s)s format: fp32:CHW
[09/01/2025-18:06:59] [I] Input build shapes: model
[09/01/2025-18:06:59] [I] Input calibration shapes: model
[09/01/2025-18:06:59] [I] === System Options ===
[09/01/2025-18:06:59] [I] Device: 0
[09/01/2025-18:06:59] [I] DLACore: 
[09/01/2025-18:06:59] [I] Plugins:
[09/01/2025-18:06:59] [I] === Inference Options ===
[09/01/2025-18:06:59] [I] Batch: 1
[09/01/2025-18:06:59] [I] Input inference shapes: model
[09/01/2025-18:06:59] [I] Iterations: 10
[09/01/2025-18:06:59] [I] Duration: 3s (+ 200ms warm up)
[09/01/2025-18:06:59] [I] Sleep time: 0ms
[09/01/2025-18:06:59] [I] Idle time: 0ms
[09/01/2025-18:06:59] [I] Streams: 1
[09/01/2025-18:06:59] [I] ExposeDMA: Disabled
[09/01/2025-18:06:59] [I] Data transfers: Enabled
[09/01/2025-18:06:59] [I] Spin-wait: Disabled
[09/01/2025-18:06:59] [I] Multithreading: Disabled
[09/01/2025-18:06:59] [I] CUDA Graph: Disabled
[09/01/2025-18:06:59] [I] Separate profiling: Disabled
[09/01/2025-18:06:59] [I] Time Deserialize: Disabled
[09/01/2025-18:06:59] [I] Time Refit: Disabled
[09/01/2025-18:06:59] [I] Skip inference: Disabled
[09/01/2025-18:06:59] [I] Inputs:
[09/01/2025-18:06:59] [I] === Reporting Options ===
[09/01/2025-18:06:59] [I] Verbose: Disabled
[09/01/2025-18:06:59] [I] Averages: 10 inferences
[09/01/2025-18:06:59] [I] Percentile: 99
[09/01/2025-18:06:59] [I] Dump refittable layers:Disabled
[09/01/2025-18:06:59] [I] Dump output: Disabled
[09/01/2025-18:06:59] [I] Profile: Disabled
[09/01/2025-18:06:59] [I] Export timing to JSON file: 
[09/01/2025-18:06:59] [I] Export output to JSON file: 
[09/01/2025-18:06:59] [I] Export profile to JSON file: 
[09/01/2025-18:06:59] [I] 
[09/01/2025-18:06:59] [I] === Device Information ===
[09/01/2025-18:06:59] [I] Selected Device: Tesla T4
[09/01/2025-18:06:59] [I] Compute Capability: 7.5
[09/01/2025-18:06:59] [I] SMs: 40
[09/01/2025-18:06:59] [I] Compute Clock Rate: 1.59 GHz
[09/01/2025-18:06:59] [I] Device Global Memory: 15109 MiB
[09/01/2025-18:06:59] [I] Shared Memory per SM: 64 KiB
[09/01/2025-18:06:59] [I] Memory Bus Width: 256 bits (ECC enabled)
[09/01/2025-18:06:59] [I] Memory Clock Rate: 5.001 GHz
[09/01/2025-18:06:59] [I] 
[09/01/2025-18:06:59] [I] TensorRT version: 8.2.5
[09/01/2025-18:06:59] [I] [TRT] [MemUsageChange] Init CUDA: CPU +336, GPU +0, now: CPU 364, GPU 252 (MiB)
[09/01/2025-18:06:59] [I] [TRT] Loaded engine size: 16 MiB
[09/01/2025-18:07:00] [W] [TRT] TensorRT was linked against cuBLAS/cuBLASLt 11.6.5 but loaded cuBLAS/cuBLASLt 11.4.1
[09/01/2025-18:07:00] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +398, GPU +166, now: CPU 769, GPU 434 (MiB)
[09/01/2025-18:07:00] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +154, GPU +172, now: CPU 923, GPU 606 (MiB)
[09/01/2025-18:07:00] [W] [TRT] TensorRT was linked against cuDNN 8.2.1 but loaded cuDNN 8.1.1
[09/01/2025-18:07:00] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +15, now: CPU 0, GPU 15 (MiB)
[09/01/2025-18:07:00] [I] Engine loaded in 1.61726 sec.
[09/01/2025-18:07:00] [W] [TRT] TensorRT was linked against cuBLAS/cuBLASLt 11.6.5 but loaded cuBLAS/cuBLASLt 11.4.1
[09/01/2025-18:07:00] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +10, now: CPU 906, GPU 598 (MiB)
[09/01/2025-18:07:00] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 906, GPU 606 (MiB)
[09/01/2025-18:07:00] [W] [TRT] TensorRT was linked against cuDNN 8.2.1 but loaded cuDNN 8.1.1
[09/01/2025-18:07:00] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +299, now: CPU 0, GPU 314 (MiB)
[09/01/2025-18:07:00] [I] Using random values for input images
[09/01/2025-18:07:00] [I] Created input binding for images with dimensions 10x3x640x1088
[09/01/2025-18:07:00] [I] Using random values for output output0
[09/01/2025-18:07:00] [I] Created output binding for output0 with dimensions 10x42840x31
[09/01/2025-18:07:00] [I] Starting inference
[09/01/2025-18:07:04] [I] Warmup completed 6 queries over 200 ms
[09/01/2025-18:07:04] [I] Timing trace has 126 queries over 3.0734 s
[09/01/2025-18:07:04] [I] 
[09/01/2025-18:07:04] [I] === Trace details ===
[09/01/2025-18:07:04] [I] Trace averages of 10 runs:
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 23.692 ms - Host latency: 34.5077 ms (end to end 47.4718 ms, enqueue 0.757256 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 23.7284 ms - Host latency: 34.5604 ms (end to end 47.2836 ms, enqueue 0.838483 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 23.7772 ms - Host latency: 34.6173 ms (end to end 47.432 ms, enqueue 0.735571 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 24.1958 ms - Host latency: 35.0158 ms (end to end 48.1289 ms, enqueue 0.728345 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 24.224 ms - Host latency: 35.0358 ms (end to end 48.4619 ms, enqueue 0.753149 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 23.5255 ms - Host latency: 34.3301 ms (end to end 45.9886 ms, enqueue 0.721301 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 22.9483 ms - Host latency: 33.7599 ms (end to end 44.8729 ms, enqueue 0.694238 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 23.7326 ms - Host latency: 34.5543 ms (end to end 47.3651 ms, enqueue 0.689099 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 24.0297 ms - Host latency: 34.8461 ms (end to end 47.9777 ms, enqueue 0.711133 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 24.44 ms - Host latency: 35.2483 ms (end to end 48.7262 ms, enqueue 0.685254 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 23.9291 ms - Host latency: 34.7744 ms (end to end 47.7793 ms, enqueue 0.686694 ms)
[09/01/2025-18:07:04] [I] Average on 10 runs - GPU latency: 23.9149 ms - Host latency: 34.7304 ms (end to end 47.6772 ms, enqueue 0.787891 ms)
[09/01/2025-18:07:04] [I] 
[09/01/2025-18:07:04] [I] === Performance summary ===
[09/01/2025-18:07:04] [I] Throughput: 40.997 qps
[09/01/2025-18:07:04] [I] Latency: min = 31.7932 ms, max = 36.7351 ms, mean = 34.6638 ms, median = 34.6692 ms, percentile(99%) = 36.0881 ms
[09/01/2025-18:07:04] [I] End-to-End Host Latency: min = 31.8066 ms, max = 51.0674 ms, mean = 47.4387 ms, median = 47.5804 ms, percentile(99%) = 50.3098 ms
[09/01/2025-18:07:04] [I] Enqueue Time: min = 0.588135 ms, max = 1.29871 ms, mean = 0.730344 ms, median = 0.708374 ms, percentile(99%) = 1.22754 ms
[09/01/2025-18:07:04] [I] H2D Latency: min = 6.76038 ms, max = 6.83435 ms, mean = 6.77499 ms, median = 6.76849 ms, percentile(99%) = 6.82928 ms
[09/01/2025-18:07:04] [I] GPU Compute Time: min = 20.9889 ms, max = 25.8992 ms, mean = 23.8437 ms, median = 23.8456 ms, percentile(99%) = 25.2888 ms
[09/01/2025-18:07:04] [I] D2H Latency: min = 4.03357 ms, max = 4.19946 ms, mean = 4.04503 ms, median = 4.04138 ms, percentile(99%) = 4.19824 ms
[09/01/2025-18:07:04] [I] Total Host Walltime: 3.0734 s
[09/01/2025-18:07:04] [I] Total GPU Compute Time: 3.00431 s
[09/01/2025-18:07:04] [I] Explanations of the performance metrics are printed in the verbose logs.
[09/01/2025-18:07:04] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8205] # trtexec --loadEngine=/home/ma-user/work/video-deal-service/weights/CP26classes_epoch_180_fp16_bs10_640_1088_fromexec.trt
