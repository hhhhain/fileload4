  写法一：auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
  // 这里把yolo层,也就是plugin的处理结果获取到,并给他取一个名字,叫做kOutputTensorName
  yolo->getOutput(0)->setName(kOutputTensorName);
  // tensorrt要求你必须显示告诉他哪几个tensor是最终输出,markOutput就是起这个作用. 这里把*yolo->getOutput(0)标记为network的最终输出
  network->markOutput(*yolo->getOutput(0));

  写法二：
          network.unmark_output(concat_tensor)
    except Exception:
        # 如果没被标记，这会抛错或返回；忽略
        pass

    # 9) 把 plugin 插入网络
    yolo_layer = network.add_plugin_v2(inputs=inputs, plugin=plugin_obj)
    if yolo_layer is None:
        raise RuntimeError("network.add_plugin_v2 returned None")

    # 10) 命名并标记输出（示例）
    yolo_layer.get_output(0).name = "yolo_out_post"
    network.mark_output(yolo_layer.get_output(0))
