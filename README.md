    # 从导出的地方载入onnx
    # onnx = file.with_suffix(".onnx")
    # 从指定的地方载入onnx
    onnx = '/home/ma-user/work/copy/yolov5_QAT-main/qat.onnx'

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
    assert onnx.exists(), f"failed to export ONNX file: {onnx}"
    f = file.with_suffix(".engine")  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
    else:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace * 1 << 30
    if cache:  # enable timing cache
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        buf = Path(cache).read_bytes() if Path(cache).exists() else b""
        timing_cache = config.create_timing_cache(buf)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f"failed to load ONNX file: {onnx}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

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
