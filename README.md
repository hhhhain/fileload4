IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT {
  // 下面的断言是做限制，第一个字段名必须是netinfo，第二个必须是kernels
  assert(fc->nbFields == 2);
  assert(strcmp(fc->fields[0].name, "netinfo") == 0);
  assert(strcmp(fc->fields[1].name, "kernels") == 0);

  // 接下来具体解析第一个字段netinfo，这是一个int数组。
  // 分别是类别数、输入宽高、每张图片最多输出多少个、是否是分割模型
  int *p_netinfo = (int*)(fc->fields[0].data);
  int class_count = p_netinfo[0];
  int input_w = p_netinfo[1];
  int input_h = p_netinfo[2];
  int max_output_object_count = p_netinfo[3];
  bool is_segmentation = (bool)p_netinfo[4];
