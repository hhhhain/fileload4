# 1) reshape -> (batch, H, W, num_anchors, per_anchor)
shuffle = network.add_shuffle(det3)
shuffle.reshape_dims = (H, W, num_anchors, per_anchor)

# 2) permute -> (batch, num_anchors, per_anchor, H, W)
shuffle.second_transpose = (3, 4, 0, 1, 2)  # 根据需要调整

# 3) 再 reshape -> (batch, num_anchors*per_anchor, H, W)
final_shuffle = network.add_shuffle(shuffle.get_output(0))
final_shuffle.reshape_dims = (num_anchors*per_anchor, H, W)
