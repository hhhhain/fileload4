    # detect模型只用了output0
    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    if dynamic:

        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
            dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            # anchors可变,指的是不一定是25200个框.
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )
