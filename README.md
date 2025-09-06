anchors = [
    [10,13, 16,30, 33,23],       # s8
    [30,61, 62,45, 59,119],      # s16
    [116,90, 156,198, 373,326]   # s32
]







# 假设 det0, det1, det2 是 IConvolutionLayer 的对象（或直接是 ITensor）
# 如果是 IConvolutionLayer，则传 det->get_output(0)，如果已经是 ITensor 则直接用
det_tensors = [det0.get_output(0), det1.get_output(0), det2.get_output(0)]
yolo_layer = add_yolo_layer_py(network, weight_map, "model.24", det_tensors=det_tensors, is_segmentation=False)



或者：
# 你之前查看到 concat 层 index 为 293
yolo_layer = add_yolo_layer_py(network, weight_map, "model.24", concat_layer_index=293, is_segmentation=False)

