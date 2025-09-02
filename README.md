ValueError: node array from the pickle has an incompatible dtype:
- expected: {'names': ['left_child', 'right_child', 'feature', 'threshold'


FP16数据
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


INT8数据：
[09/02/2025-18:08:30] [I] Trace averages of 10 runs:
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 24.1244 ms - Host latency: 34.939 ms (enqueue 1.00458 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.6688 ms - Host latency: 31.4901 ms (enqueue 1.05826 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.6889 ms - Host latency: 31.5102 ms (enqueue 1.0718 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.5939 ms - Host latency: 31.4242 ms (enqueue 1.17789 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 21.2726 ms - Host latency: 32.0896 ms (enqueue 1.00988 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 21.0987 ms - Host latency: 31.9158 ms (enqueue 0.955457 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.8198 ms - Host latency: 31.6389 ms (enqueue 0.919067 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.6765 ms - Host latency: 31.4951 ms (enqueue 0.91698 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.7342 ms - Host latency: 31.551 ms (enqueue 0.901587 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 21.1856 ms - Host latency: 32.0014 ms (enqueue 0.931372 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 21.0309 ms - Host latency: 31.8491 ms (enqueue 1.0562 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.8226 ms - Host latency: 31.6417 ms (enqueue 0.97002 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.8075 ms - Host latency: 31.6456 ms (enqueue 1.2073 ms)
[09/02/2025-18:08:30] [I] Average on 10 runs - GPU latency: 20.7532 ms - Host latency: 31.5878 ms (enqueue 1.09297 ms)



fp16+sp90：
[09/02/2025-18:25:43] [I] === Trace details ===
[09/02/2025-18:25:43] [I] Trace averages of 10 runs:
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 7.58589 ms - Host latency: 20.8187 ms (enqueue 0.94856 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 6.58893 ms - Host latency: 20.5428 ms (enqueue 0.958078 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.85572 ms - Host latency: 20.3958 ms (enqueue 0.953769 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.49644 ms - Host latency: 20.301 ms (enqueue 0.924597 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.49514 ms - Host latency: 20.3142 ms (enqueue 1.01356 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.52186 ms - Host latency: 20.3375 ms (enqueue 1.07581 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.52049 ms - Host latency: 20.324 ms (enqueue 1.07987 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.48647 ms - Host latency: 20.3116 ms (enqueue 1.032 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.4694 ms - Host latency: 20.3105 ms (enqueue 1.29401 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.46011 ms - Host latency: 20.3023 ms (enqueue 0.983606 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.4964 ms - Host latency: 20.3047 ms (enqueue 1.09485 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.51022 ms - Host latency: 20.3443 ms (enqueue 1.02393 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.49556 ms - Host latency: 20.3104 ms (enqueue 0.990759 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.48737 ms - Host latency: 20.2995 ms (enqueue 0.899451 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.46708 ms - Host latency: 20.298 ms (enqueue 0.91532 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.4646 ms - Host latency: 20.293 ms (enqueue 0.908911 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.49113 ms - Host latency: 20.3207 ms (enqueue 1.06523 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.5068 ms - Host latency: 20.3303 ms (enqueue 1.11981 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.53848 ms - Host latency: 20.3332 ms (enqueue 1.03264 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.5052 ms - Host latency: 20.3003 ms (enqueue 0.971167 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.48513 ms - Host latency: 20.2916 ms (enqueue 0.901685 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.46533 ms - Host latency: 20.3118 ms (enqueue 0.907104 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.46506 ms - Host latency: 20.3073 ms (enqueue 0.937305 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.4929 ms - Host latency: 20.3075 ms (enqueue 0.96936 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.53257 ms - Host latency: 20.3102 ms (enqueue 0.912354 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.51177 ms - Host latency: 20.3029 ms (enqueue 0.89209 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.49602 ms - Host latency: 20.2989 ms (enqueue 0.911206 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.48411 ms - Host latency: 20.3029 ms (enqueue 0.939917 ms)
[09/02/2025-18:25:43] [I] Average on 10 runs - GPU latency: 5.46353 ms - Host latency: 20.3003 ms (enqueue 0.924829 ms)
[09/02/2025-18:25:43] [I] 
[09/02/2025-18:25:43] [I] === Performance summary ===
[09/02/2025-18:25:43] [I] Throughput: 97.4735 qps
[09/02/2025-18:25:43] [I] Latency: min = 17.9661 ms, max = 21.3447 ms, mean = 20.3304 ms, median = 20.3074 ms, percentile(90%) = 20.3759 ms, percentile(95%) = 20.5475 ms, percentile(99%) = 21.3119 ms
[09/02/2025-18:25:43] [I] Enqueue Time: min = 0.80957 ms, max = 2.01843 ms, mean = 0.984369 ms, median = 0.931885 ms, percentile(90%) = 1.1709 ms, percentile(95%) = 1.34851 ms, percentile(99%) = 1.69519 ms
[09/02/2025-18:25:43] [I] H2D Latency: min = 6.96774 ms, max = 8.96735 ms, mean = 8.63143 ms, median = 8.67065 ms, percentile(90%) = 8.76855 ms, percentile(95%) = 8.81665 ms, percentile(99%) = 8.95279 ms
[09/02/2025-18:25:43] [I] GPU Compute Time: min = 5.26343 ms, max = 9.52582 ms, mean = 5.61187 ms, median = 5.4939 ms, percentile(90%) = 5.55164 ms, percentile(95%) = 6.5802 ms, percentile(99%) = 9.51146 ms
[09/02/2025-18:25:43] [I] D2H Latency: min = 4.03418 ms, max = 6.45227 ms, mean = 6.08705 ms, median = 6.1366 ms, percentile(90%) = 6.23645 ms, percentile(95%) = 6.28351 ms, percentile(99%) = 6.42758 ms
[09/02/2025-18:25:43] [I] Total Host Walltime: 3.02646 s
[09/02/2025-18:25:43] [I] Total GPU Compute Time: 1.6555 s
[09/02/2025-18:25:43] [I] Explanations of the performance metrics are printed in the verbose logs.
[09/02/2025-18:25:43] [V] 
[09/02/2025-18:25:43] [V] === Explanations of the performance metrics ===
[09/02/2025-18:25:43] [V] Total Host Walltime: the host walltime from when the first query (after warmups) is enqueued to when the last query is completed.
[09/02/2025-18:25:43] [V] GPU Compute Time: the GPU latency to execute the kernels for a query.
[09/02/2025-18:25:43] [V] Total GPU Compute Time: the summation of the GPU Compute Time of all the queries. If this is significantly shorter than Total Host Walltime, the GPU may be under-utilized because of host-side overheads or data transfers.
[09/02/2025-18:25:43] [V] Throughput: the observed throughput computed by dividing the number of queries by the Total Host Walltime. If this is significantly lower than the reciprocal of GPU Compute Time, the GPU may be under-utilized because of host-side overheads or data transfers.
[09/02/2025-18:25:43] [V] Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.
[09/02/2025-18:25:43] [V] H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.
[09/02/2025-18:25:43] [V] D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.
[09/02/2025-18:25:43] [V] Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a single query.
[09/02/2025-18:25:43] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8601] # trtexec --loadEngine=/home/ma-user/work/copy/files/video-deal-search/video-deal-service/weights/CP26classes_epoch_180_fp16_bs10_640_1088_sp0.9.trt --verbose


fp16+sp40：
[09/02/2025-18:40:29] [I] === Trace details ===
[09/02/2025-18:40:29] [I] Trace averages of 10 runs:
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 24.603 ms - Host latency: 35.4135 ms (enqueue 1.0513 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.517 ms - Host latency: 32.3273 ms (enqueue 1.07826 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.4466 ms - Host latency: 32.2604 ms (enqueue 1.06891 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.709 ms - Host latency: 32.5199 ms (enqueue 1.09145 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 22.6138 ms - Host latency: 33.4169 ms (enqueue 1.1835 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 22.06 ms - Host latency: 32.8737 ms (enqueue 1.11478 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.7185 ms - Host latency: 32.5305 ms (enqueue 1.04695 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.5241 ms - Host latency: 32.3383 ms (enqueue 1.03336 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.8959 ms - Host latency: 32.7065 ms (enqueue 1.04111 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 22.388 ms - Host latency: 33.195 ms (enqueue 1.03955 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.914 ms - Host latency: 32.727 ms (enqueue 1.03083 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.9366 ms - Host latency: 32.7506 ms (enqueue 1.05386 ms)
[09/02/2025-18:40:29] [I] Average on 10 runs - GPU latency: 21.9139 ms - Host latency: 32.7263 ms (enqueue 1.06414 ms)
[09/02/2025-18:40:29] [I] 
[09/02/2025-18:40:29] [I] === Performance summary ===
[09/02/2025-18:40:29] [I] Throughput: 44.7524 qps
[09/02/2025-18:40:29] [I] Latency: min = 31.7702 ms, max = 46.0043 ms, mean = 32.8976 ms, median = 32.6096 ms, percentile(90%) = 33.4585 ms, percentile(95%) = 34.1327 ms, percentile(99%) = 45.9589 ms
[09/02/2025-18:40:29] [I] Enqueue Time: min = 0.95575 ms, max = 1.56281 ms, mean = 1.06752 ms, median = 1.04944 ms, percentile(90%) = 1.10327 ms, percentile(95%) = 1.26672 ms, percentile(99%) = 1.4884 ms
[09/02/2025-18:40:29] [I] H2D Latency: min = 6.76003 ms, max = 6.79333 ms, mean = 6.7713 ms, median = 6.76971 ms, percentile(90%) = 6.77979 ms, percentile(95%) = 6.78485 ms, percentile(99%) = 6.79053 ms
[09/02/2025-18:40:29] [I] GPU Compute Time: min = 20.9649 ms, max = 35.201 ms, mean = 22.0864 ms, median = 21.7998 ms, percentile(90%) = 22.6484 ms, percentile(95%) = 23.3313 ms, percentile(99%) = 35.1629 ms
[09/02/2025-18:40:29] [I] D2H Latency: min = 4.03223 ms, max = 4.06714 ms, mean = 4.03995 ms, median = 4.03876 ms, percentile(90%) = 4.047 ms



fp16+sp50：
[09/02/2025-18:44:08] [I] === Trace details ===
[09/02/2025-18:44:08] [I] Trace averages of 10 runs:
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 16.2002 ms - Host latency: 27.046 ms (enqueue 0.862134 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8686 ms - Host latency: 22.7902 ms (enqueue 0.807352 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8415 ms - Host latency: 22.7632 ms (enqueue 0.796741 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8438 ms - Host latency: 22.7611 ms (enqueue 0.772632 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8065 ms - Host latency: 22.7278 ms (enqueue 0.802991 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8373 ms - Host latency: 22.7599 ms (enqueue 0.832422 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8234 ms - Host latency: 22.7442 ms (enqueue 0.797418 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.897 ms - Host latency: 22.8162 ms (enqueue 0.782288 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 12.3657 ms - Host latency: 23.2834 ms (enqueue 0.918469 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 12.1063 ms - Host latency: 23.0275 ms (enqueue 0.923059 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.9292 ms - Host latency: 22.8484 ms (enqueue 0.934009 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8562 ms - Host latency: 22.7821 ms (enqueue 0.904041 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8492 ms - Host latency: 22.7789 ms (enqueue 0.898914 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8333 ms - Host latency: 22.7556 ms (enqueue 0.850049 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8355 ms - Host latency: 22.7576 ms (enqueue 0.863684 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8901 ms - Host latency: 22.8084 ms (enqueue 0.813403 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 12.1707 ms - Host latency: 23.0886 ms (enqueue 0.818799 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 12.1652 ms - Host latency: 23.0823 ms (enqueue 0.779297 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.9858 ms - Host latency: 22.9049 ms (enqueue 0.781592 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.9028 ms - Host latency: 22.831 ms (enqueue 0.842358 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8911 ms - Host latency: 22.8221 ms (enqueue 0.827441 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8488 ms - Host latency: 22.7761 ms (enqueue 0.972266 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.838 ms - Host latency: 22.7591 ms (enqueue 0.797729 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 11.8565 ms - Host latency: 22.7801 ms (enqueue 0.791919 ms)
[09/02/2025-18:44:08] [I] Average on 10 runs - GPU latency: 12.0602 ms - Host latency: 22.9665 ms (enqueue 0.870728 ms)
[09/02/2025-18:44:08] [I] 
[09/02/2025-18:44:08] [I] === Performance summary ===
[09/02/2025-18:44:08] [I] Throughput: 82.1463 qps
[09/02/2025-18:44:08] [I] Latency: min = 22.5546 ms, max = 28.5039 ms, mean = 23.0184 ms, median = 22.784 ms, percentile(90%) = 23.1975 ms, percentile(95%) = 23.2803 ms, percentile(99%) = 28.4885 ms
[09/02/2025-18:44:08] [I] Enqueue Time: min = 0.708252 ms, max = 1.4314 ms, mean = 0.841669 ms, median = 0.818832 ms, percentile(90%) = 0.904785 ms, percentile(95%) = 1.1228 ms, percentile(99%) = 1.2771 ms
[09/02/2025-18:44:08] [I] H2D Latency: min = 6.76492 ms, max = 6.79785 ms, mean = 6.77394 ms, median = 6.7728 ms, percentile(90%) = 6.77979 ms, percentile(95%) = 6.78345 ms, percentile(99%) = 6.78882 ms
[09/02/2025-18:44:08] [I] GPU Compute Time: min = 11.6428 ms, max = 17.6765 ms, mean = 12.1001 ms, median = 11.8583 ms, percentile(90%) = 12.2859 ms, percentile(95%) = 12.3678 ms, percentile(99%) = 17.6723 ms
[09/02/2025-18:44:08] [I] D2H Latency: min = 4.03442 ms, max = 4.16797 ms, mean = 4.14439 ms, median = 4.14691 ms, percentile(90%)














[09/02/2025-19:47:56] [I] === Trace details ===
[09/02/2025-19:47:56] [I] Trace averages of 10 runs:
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 15.5601 ms - Host latency: 26.3838 ms (enqueue 0.949657 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 14.0518 ms - Host latency: 24.8892 ms (enqueue 0.864416 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.6997 ms - Host latency: 24.5386 ms (enqueue 0.825217 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.6882 ms - Host latency: 24.5278 ms (enqueue 0.851898 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.7396 ms - Host latency: 24.5801 ms (enqueue 0.85885 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.6886 ms - Host latency: 24.5279 ms (enqueue 0.83916 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.735 ms - Host latency: 24.5747 ms (enqueue 0.896716 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.9485 ms - Host latency: 24.7948 ms (enqueue 1.07068 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 14.0508 ms - Host latency: 24.8968 ms (enqueue 1.0229 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.7523 ms - Host latency: 24.6057 ms (enqueue 1.10677 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.739 ms - Host latency: 24.5802 ms (enqueue 1.00106 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.8468 ms - Host latency: 24.6857 ms (enqueue 0.890552 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.8333 ms - Host latency: 24.6718 ms (enqueue 0.839343 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.8108 ms - Host latency: 24.6525 ms (enqueue 0.884302 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 14.0035 ms - Host latency: 24.8416 ms (enqueue 0.877197 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.9882 ms - Host latency: 24.8222 ms (enqueue 0.79502 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.8921 ms - Host latency: 24.7266 ms (enqueue 0.819336 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.8101 ms - Host latency: 24.6513 ms (enqueue 0.886719 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.8582 ms - Host latency: 24.6974 ms (enqueue 0.869409 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.845 ms - Host latency: 24.6818 ms (enqueue 0.866309 ms)
[09/02/2025-19:47:56] [I] Average on 10 runs - GPU latency: 13.8224 ms - Host latency: 24.6628 ms (enqueue 0.862598 ms)
[09/02/2025-19:47:56] [I] 
[09/02/2025-19:47:56] [I] === Performance summary ===
[09/02/2025-19:47:56] [I] Throughput: 71.3456 qps
[09/02/2025-19:47:56] [I] Latency: min = 24.3962 ms, max = 31.0309 ms, mean = 24.7604 ms, median = 24.6741 ms, percentile(90%) = 24.9083 ms, percentile(95%) = 25.0825 ms, percentile(99%) = 25.9111 ms
[09/02/2025-19:47:56] [I] Enqueue Time: min = 0.736572 ms, max = 1.86707 ms, mean = 0.898283 ms, median = 0.86377 ms, percentile(90%) = 0.991699 ms, percentile(95%) = 1.21057 ms, percentile(99%) = 1.36786 ms
[09/02/2025-19:47:56] [I] H2D Latency: min = 6.7637 ms, max = 6.81177 ms, mean = 6.77635 ms, median = 6.77515 ms, percentile(90%) = 6.78235 ms, percentile(95%) = 6.78754 ms, percentile(99%) = 6.80042 ms
[09/02/2025-19:47:56] [I] GPU Compute Time: min = 13.5906 ms, max = 20.2181 ms, mean = 13.9212 ms, median = 13.8342 ms, percentile(90%) = 14.0698 ms, percentile(95%) = 14.2562 ms, percentile(99%) = 15.0935 ms
[09/02/2025-19:47:56] [I] D2H Latency: min = 4.03564 ms, max = 4.07367 ms, mean = 4.06291 ms, median = 4.06323 ms, percentile(90%) = 4.06732 ms, percentile(95%) = 4.06796 ms, percentile(99%) = 4.07227 ms
[09/02/2025-19:47:56] [I] Total Host Walltime: 3.04153 s
[09/02/2025-19:47:56] [I] Total GPU Compute Time: 3.0209 s
[09/02/2025-19:47:56] [I] Explanations of the performance metrics are printed in the verbose logs.
[09/02/2025-19:47:56] [V] 
[09/02/2025-19:47:56] [V] === Explanations of the performance metrics ===
[09/02/2025-19:47:56] [V] Total Host Walltime: the host walltime from when the first query (after warmups) is enqueued to when the last query is completed.
[09/02/2025-19:47:56] [V] GPU Compute Time: the GPU latency to execute the kernels for a query.
[09/02/2025-19:47:56] [V] Total GPU Compute Time: the summation of the GPU Compute Time of all the queries. If this is significantly shorter than Total Host Walltime, the GPU may be under-utilized because of host-side overheads or data transfers.
[09/02/2025-19:47:56] [V] Throughput: the observed throughput computed by dividing the number of queries by the Total Host Walltime. If this is significantly lower than the reciprocal of GPU Compute Time, the GPU may be under-utilized because of host-side overheads or data transfers.
[09/02/2025-19:47:56] [V] Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.
[09/02/2025-19:47:56] [V] H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.
[09/02/2025-19:47:56] [V] D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.
[09/02/2025-19:47:56] [V] Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a single query.
[09/02/2025-19:47:56] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8601] # trtexec --loadEngine=/home/ma-user/work/copy/files/video-deal-search/video-deal-service/weights/CP26classes_epoch_180_int8_bs10_640_1088_sp0.5.trt --verbose
