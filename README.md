        ctypes.CDLL("/home/ma-user/work/myplugin/build/libAddOnePlugin.so", mode=ctypes.RTLD_GLOBAL)
        # 初始化Plugin插件, 也可以提供一个logger替换None,记录信息
        trt.init_libnvinfer_plugins(None, "")
