# 假设你已经有 plugin 对象 plugin
# 获取 YOLOv5 detection 输出的三个特征层
det_out_s8 = network.get_output(0)
det_out_s16 = network.get_output(1)
det_out_s32 = network.get_output(2)

# 先取消原始输出
for i in range(3):
    network.unmark_output(network.get_output(i))

# 对每个 detection 层单独插入 plugin
plugin_s8 = network.add_plugin_v2([det_out_s8], plugin)
plugin_s8.get_output(0).name = "s8_out_original"
plugin_s8.get_output(1).name = "s8_out_plus_one"

plugin_s16 = network.add_plugin_v2([det_out_s16], plugin)
plugin_s16.get_output(0).name = "s16_out_original"
plugin_s16.get_output(1).name = "s16_out_plus_one"

plugin_s32 = network.add_plugin_v2([det_out_s32], plugin)
plugin_s32.get_output(0).name = "s32_out_original"
plugin_s32.get_output(1).name = "s32_out_plus_one"

# 标记为网络输出
for layer in [plugin_s8, plugin_s16, plugin_s32]:
    network.mark_output(layer.get_output(0))
    network.mark_output(layer.get_output(1))
