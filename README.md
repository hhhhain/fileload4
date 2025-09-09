dynamic = {
    # 输入: (N,3,H,W) → batch/H/W 都动态
    "images": {0: "batch", 2: "height", 3: "width"},

    # 输出: (N,anchors,85) → batch/anchors 都动态
    "output0": {0: "batch", 1: "anchors"}
}

quantize.export_onnx(
    model,
    dummy,
    file,
    opset_version=17,
    input_names=["images"],
    output_names=["output0"],
    dynamic_axes=dynamic
)
