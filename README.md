network.unmark_output(network.get_output(0))  # 先取消原始输出

last_tensor = network.get_layer(network.num_layers - 1).get_output(0)

# 把 plugin 插进去
plugin_layer = network.add_plugin_v2([last_tensor], plugin)

# Plugin 有两个输出：原始 + 加1
plugin_layer.get_output(0).name = "out_original"
plugin_layer.get_output(1).name = "out_plus_one"

# 重新标记为网络输出
network.mark_output(plugin_layer.get_output(0))
network.mark_output(plugin_layer.get_output(1))
