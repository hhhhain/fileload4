        quantize.export_onnx(model, dummy, file, opset_version=17, 
            input_names=["images"], output_names=["s8", "s16", "s32"], 
            dynamic_axes={"images": {0: "batch"}, "s32": {0: "batch"}, "s16": {0: "batch"}, "s8": {0: "batch"}} if dynamic_batch else None
        )

        

                dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
            dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            # anchors可变,指的是不一定是25200个框.
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
