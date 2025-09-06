    # 加载Plugin的示例代码↓↓↓↓↓↓↓↓↓↓↓↓↓↓初始化部分↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # 加载Plugin的动态链接库
    ctypes.CDLL("/home/ma-user/work/myplugin/build/libAddOnePlugin.so", mode=ctypes.RTLD_GLOBAL)
    # 初始化Plugin插件, 也可以提供一个logger替换None,记录信息
    trt.init_libnvinfer_plugins(None, "")
    # 获得Plugin列表.
    registry = trt.get_plugin_registry()
    # 从列表里找到指定的Plugin创建器.
    creator = registry.get_plugin_creator("AddOnePlugin", "1", "")
    if creator is None:
        raise RuntimeError("AddOnePlugin is not foune in registry!")
    # 加载Plugin的示例代码↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑初始化部分↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    concat_out = network.get_layer(293).get_output(0)
    network.unmark_output(concat_out)

    plugin_layer = network.add_plugin_v2([concat_out], plugin)
    plugin_layer.get_output(0).name = "out_original"
    plugin_layer.get_output(1).name = "out_plus_one"

    network.mark_output(plugin_layer.get_output(0))
    network.mark_output(plugin_layer.get_output(1))
