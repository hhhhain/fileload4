concat_out = network.get_layer(293).get_output(0)
network.unmark_output(concat_out)

plugin_layer = network.add_plugin_v2([concat_out], plugin)
plugin_layer.get_output(0).name = "out_original"
plugin_layer.get_output(1).name = "out_plus_one"

network.mark_output(plugin_layer.get_output(0))
network.mark_output(plugin_layer.get_output(1))
