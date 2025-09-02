        onnx_path = save_qat.replace(".pt", ".onnx")

        export_onnx(model, onnx_path, 640)
        print(f"Exported QAT ONNX model to {onnx_path}")     
        


    
        import tensorrt as trt
        onnx_file = Path(onnx_path)

        # LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
        assert onnx_file.exists(), f"failed to export ONNX file: {onnx_file}"
        f = onnx_file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        # if verbose:
        #     logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        else:  # TensorRT versions 7, 8
            config.max_workspace_size = 4 * 1 << 30
        # if cache:  # enable timing cache
        #     Path(cache).parent.mkdir(parents=True, exist_ok=True)
        #     buf = Path(cache).read_bytes() if Path(cache).exists() else b""
        #     timing_cache = config.create_timing_cache(buf)
        #     config.set_timing_cache(timing_cache, ignore_mismatch=True)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        # if not parser.parse_from_file(str(onnx_file)):
        #     raise RuntimeError(f"failed to load ONNX file: {onnx_file}")
