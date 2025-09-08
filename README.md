    det1 = network.get_layer(242).get_output(0)
    det2 = network.get_layer(267).get_output(0)
    det3 = network.get_layer(292).get_output(0)
    add_yolo_layer_py(network, det_tensors=[det1,det2,det3], concat_layer_index=293, is_segmentation=False)


    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    # exit()

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    # exit()
    with build(network, config) as engine, open(f, "wb") as t:
        t.write(engine if is_trt10 else engine.serialize())
    if cache:  # save timing cache
        with open(cache, "wb") as c:
            c.write(config.get_timing_cache().serialize())
    return f, None
