static std::vector<std::vector<float>> getAnchors(std::map<std::string, Weights>& weightMap, std::string lname) {
  std::vector<std::vector<float>> anchors;
  Weights wts = weightMap[lname + ".anchor_grid"];
  int anchor_len = kNumAnchor * 2;
  for (int i = 0; i < wts.count / anchor_len; i++) {
    auto *p = (const float*)wts.values + i * anchor_len;
    std::vector<float> anchor(p, p + anchor_len);
    anchors.push_back(anchor);
  }
  return anchors;
}







# 假设 det0, det1, det2 是 IConvolutionLayer 的对象（或直接是 ITensor）
# 如果是 IConvolutionLayer，则传 det->get_output(0)，如果已经是 ITensor 则直接用
det_tensors = [det0.get_output(0), det1.get_output(0), det2.get_output(0)]
yolo_layer = add_yolo_layer_py(network, weight_map, "model.24", det_tensors=det_tensors, is_segmentation=False)



或者：
# 你之前查看到 concat 层 index 为 293
yolo_layer = add_yolo_layer_py(network, weight_map, "model.24", concat_layer_index=293, is_segmentation=False)

