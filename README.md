for i in range(network.num_layers):
    layer = network.get_layer(i)
    print(f"index={i}, name={layer.name}, type={layer.type}, nb_outputs={layer.num_outputs}")
    for j in range(layer.num_outputs):
        print(f"  output[{j}] shape: {layer.get_output(j).shape}")
