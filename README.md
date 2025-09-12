        PLUGIN_LIBRARY = "packages/libmyplugins.so"
        ctypes.CDLL(PLUGIN_LIBRARY)
        PLUGIN_LIBRARY2 = "packages/libv8myplugins_v2.so"
        ctypes.CDLL(PLUGIN_LIBRARY2)
        cp_yolov5trt = YoLov5TRT(cp_engine_file_path)
        screen_yolov5trt = YoLov5TRT(screen_engine_file_path)
        pose_yolov5trt = YoLov5TRT(pose_engine_file_path)
