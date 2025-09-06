# 假设你已经有 plugin 对象 plugin
# 获取 YOLOv5 detection 输出的三个特征层
det_out_s8 = network.get_output(0)
det_out_s16 = network.get_output(1)
det_out_s32 = network.get_output(2)

# 先取消原始输出
for i in range(3):
    network.unmark_output(network.get_output(i))

# 插入 plugin，输入是三个 detection 输出
plugin_layer = network.add_plugin_v2([det_out_s8, det_out_s16, det_out_s32], plugin)

# plugin 有两个输出：原始 + 加1
plugin_layer.get_output(0).name = "out_original"
plugin_layer.get_output(1).name = "out_plus_one"

# 标记为网络输出
network.mark_output(plugin_layer.get_output(0))
network.mark_output(plugin_layer.get_output(1))
