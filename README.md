for t in det_tensors:
    if t.is_network_output:  # 确认这个 tensor 真的被标记为输出
        network.unmark_output(t)
