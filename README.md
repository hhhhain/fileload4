__global__ void CalDetection(const float *input, float *output, int noElements,
    const int netwidth, const int netheight, int maxoutobject, int yoloWidth,
    int yoloHeight, const float anchors[kNumAnchor * 2], int classes, int outputElem, bool is_segmentation) {
// 参数说明：
// input，网络输出的整个batch的feature map，batch，chnnls，h，w的指针。
// output，存储Detection结构的输出buffer
// noElements，总网格点数=yoloW * yoloH * batchsize
// anchors，当前head的anchors
// classes，类别数。
// outputElem，单张图的输出float数量。
// is_seg，是否包含mask分支。
// threadIdx、blockDim、blockIdx并不是自己定义的变量，而是CUDA自动提供的内置变量，在每个线程执行kernel时gpu自动赋值。
// 这里有个知识点，idx已经跟当前线程处理的grid cell绑定了，所以下面的循环不再需要idx层面循环。
// 这里的意思是每个线程处理一个grid cell，比如输出特征图是20x20，那一共有400个grid cell。
// 所以下面的变量其实都是有变化的，比如bnIdx = idx / total_grid，他会首先判断idx是哪一个，对应哪一个cell。
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= noElements) return; 

  int total_grid = yoloWidth * yoloHeight;
  int bnIdx = idx / total_grid; // batch的索引，相当于第几张图，10张图的第几张。
  idx = idx - total_grid * bnIdx;// 是偏移。相对于当前batch图像的起始位置的网格索引。
  int info_len_i = 5 + classes; //每个anchor的元素数量，这个好理解，比如5+80.5是由boss，conf组成。
  if (is_segmentation) info_len_i += 32; //分割。
  // kNumAnchor如前文猜测应该是每个head的anchor数，乘以格子数，乘以每个格子的信息量，再乘以第几张图，然后对input指针进行偏移，指向某一张图的起始位置。
  const float* curInput = input + bnIdx * (info_len_i * total_grid * kNumAnchor); 
  // kNumAnchor如前文猜测应该是每个head的anchor数，每个cell同样，比如是3. 遍历每个anchor， 
  for (int k = 0; k < kNumAnchor; ++k) {
    // 这条语句是在对某一张图的所有cell求置信度，0或者1. 为什么是所有cell？因为每一个idx都绑定了线程，都是并行执行的。[anchor1][anchor2][anchor3]的顺序在内存中存放。
    float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
    // 所有cell都是并行的，所以如果判断出当前的cell的置信度太低的话，就不管这个cell了。执行下一次for，也就是下一个anchor，同理。
    if (box_prob < kIgnoreThresh) continue;
    int class_id = 0;
    float max_cls_prob = 0.0;
    // 遍历80个类，找到最大的可能性，就是class。同样是每个cell并行的。
    for (int i = 5; i < 5 + classes; ++i) {
      float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
      if (p > max_cls_prob) {
        max_cls_prob = p;
        class_id = i - 5;
      }
    }
    // 第几张乘以单张的大小，res_count指当前这一张的输出起点。第一位res_count[0]记录了已写入的数量。
    // atomicAdd是原子加法，防止多线程写入同一个地址造成覆盖等。res_count总数+1，比如已经写了1223个detection结果。detection是上面说的结构体：坐标、置信度、类别。
    // 这里有个问题，这里的超过框的总数就不写了，是怎么一个并行情况？都在抢着写？怎么一个层次去抢的？初步感觉是所有的cell都在判断最小的anchor，从小到大了写。
    float *res_count = output + bnIdx * outputElem;
    int count = (int)atomicAdd(res_count, 1);
    if (count >= maxoutobject) return;
    // 每个cell计算自己应该要写的位置，转成detection指针，以后要写的时候就可以用det->结构。比如下面。
    char *data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
    Detection *det = (Detection*)(data);

    // 定位行和列。
    int row = idx / yoloWidth;
    int col = idx % yoloWidth;
    
    // 定位中心坐标。其实就是col/row - 0.5 + 2 * sigmoid。而sigmoid在01之间，所以col/row + (-0.5~1.5), 所以是相对于格子左上角，可以左偏0.5格子到右偏1.5个格子。然后* netwidth / yoloWidth是转换回原图坐标系。
    // 这里计算的是框中心x和y。
    det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
    det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

    // 2.0f * sigmoid映射到0到2之间。
    det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
    // 最大是4倍anchor，最小当然是就是0了，靠sigmoid调控。宽高同理。
    // 比较重要的是为什么中心点* netwidth / yoloWidth映射回去了，而宽高不需要映射。因为anchor box本身就是在输入特征图大小考虑的，比如640，640. 多少倍的anchor长宽，得到的框数据就是640*640的原图长宽。
    // 那为什么中心点需要映射回去呢？因为前向传播推理得到的中心点是基于输出特征图的cell的，不是基于640 640这个输入特征图的。所以要把中心点从输出特征图的cell映射回640 640这个输入特征图。
    det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
    det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
    det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];

    // 有物体的置信度乘以这个物体是啥的分类得分，就是联合置信度conf。
    det->conf = box_prob * max_cls_prob;
    det->class_id = class_id;

    for (int i = 0; is_segmentation && i < 32; i++) {
      det->mask[i] = curInput[idx + k * info_len_i * total_grid + (i + 5 + classes) * total_grid];
    }
  }
}
  
  
  
  
  
  
  // kNumClass, kInputW, kInputH, kMaxNumOutputBbox这几个参数在config.h头文件里宏定义.
  int netinfo[5] = {kNumClass, kInputW, kInputH, kMaxNumOutputBbox, (int)is_segmentation};
  plugin_fields[0].data = netinfo;
  plugin_fields[0].length = 5;
  plugin_fields[0].name = "netinfo";
  plugin_fields[0].type = PluginFieldType::kFLOAT32;

  //load strides from Detect layer
  assert(weightMap.find(lname + ".strides") != weightMap.end() && "Not found `strides`, please check gen_wts.py!!!");
  Weights strides = weightMap[lname + ".strides"];
  auto *p = (const float*)(strides.values);
  std::vector<int> scales(p, p + strides.count);

  std::vector<YoloKernel> kernels;
  for (size_t i = 0; i < anchors.size(); i++) {
    YoloKernel kernel;
    kernel.width = kInputW / scales[i];
    kernel.height = kInputH / scales[i];
    memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
    // push_back用来在vector末尾添加一个元素.
    kernels.push_back(kernel);
  }
  plugin_fields[1].data = &kernels[0];
  plugin_fields[1].length = kernels.size();
  plugin_fields[1].name = "kernels";
  plugin_fields[1].type = PluginFieldType::kFLOAT32;
  PluginFieldCollection plugin_data;
  plugin_data.nbFields = 2;
  plugin_data.fields = plugin_fields;



  





import tensorrt as trt
import numpy as np

def add_yolo_layer(network, weight_map, lname, dets, is_segmentation=False):
    # 1. 获取插件 creator
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("YoloLayer_TRT", "1")
    if creator is None:
        raise RuntimeError("YoloLayer_TRT plugin not found in registry!")

    # 2. 构造 netinfo
    netinfo = np.array(
        [kNumClass, kInputW, kInputH, kMaxNumOutputBbox, int(is_segmentation)],
        dtype=np.float32
    )
    netinfo_field = trt.PluginField(
        name="netinfo",
        data=netinfo,
        type=trt.PluginFieldType.FLOAT32
    )

    # 3. 加载 strides
    if lname + ".strides" not in weight_map:
        raise RuntimeError("Not found strides, please check gen_wts.py!!!")
    strides = weight_map[lname + ".strides"]  # 这里通常是 Weights
    scales = np.array(strides, dtype=np.float32)  # strides.values -> numpy

    # 4. 构造 kernels
    kernels = []
    anchors = getAnchors(weight_map, lname)  # 需要你实现 getAnchors()
    for i in range(len(anchors)):
        w = kInputW / scales[i]
        h = kInputH / scales[i]
        # 一个 YoloKernel = (width, height, anchors)
        kernel = [w, h] + anchors[i]
        kernels.extend(kernel)
    kernels = np.array(kernels, dtype=np.float32)

    kernels_field = trt.PluginField(
        name="kernels",
        data=kernels,
        type=trt.PluginFieldType.FLOAT32
    )

    # 5. 封装 PluginFieldCollection
    field_collection = trt.PluginFieldCollection([netinfo_field, kernels_field])

    # 6. 创建 plugin
    plugin_obj = creator.create_plugin("yololayer", field_collection)

    # 7. 收集 det 层输出作为输入
    input_tensors = [det.get_output(0) for det in dets]

    # 8. 插入 plugin
    yolo_layer = network.add_plugin_v2(inputs=input_tensors, plugin=plugin_obj)
    return yolo_layer
