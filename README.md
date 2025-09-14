import numpy as np

# det3 是 ONNX 输出 (batch, num_boxes, 31)
batch, num_boxes, c = det3.shape

# 你的 detection 层有 3 anchors
num_anchors = 3
num_classes = 26
per_anchor = num_classes + 5  # 31

# 确认 c == per_anchor
assert c == per_anchor

# 计算特征图尺寸 H, W
# det3.num_boxes = H * W * num_anchors
H = W = int((num_boxes // num_anchors) ** 0.5)
assert H * W * num_anchors == num_boxes

# reshape 回卷积输出 (batch, num_anchors*(num_classes+5), H, W)
det3_conv_like = det3.reshape(batch, H, W, num_anchors, per_anchor)  # (B,H,W,3,31)
det3_conv_like = det3_conv_like.transpose(0, 3, 4, 1, 2)             # (B,3,31,H,W)
det3_conv_like = det3_conv_like.reshape(batch, num_anchors * per_anchor, H, W)  # (B,93,H,W)
