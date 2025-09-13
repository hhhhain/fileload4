#include "yololayer.h"
#include "cuda_utils.h"

#include <cassert>
#include <vector>
#include <iostream>

namespace Tn {
template<typename T> 
void write(char*& buffer, const T& val) {
  *reinterpret_cast<T*>(buffer) = val;
  buffer += sizeof(T);
}

template<typename T> 
void read(const char*& buffer, T& val) {
  val = *reinterpret_cast<const T*>(buffer);
  buffer += sizeof(T);
}
}

namespace nvinfer1 {
// 最终获得了plugin的成员变量。以及GPU里面存入了：
// mAnchor[0] > 指向GPU内存，里面是 ([10,13],[16,30],[33,23])
// mAnchor[1] > 指向GPU内存，里面是 ([30,61],[62,45],[59,119])
// mAnchor[2] > 指向GPU内存，里面是 ([116,90],[156,198],[373,326])
YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, bool is_segmentation, const std::vector<YoloKernel>& vYoloKernel) {
  mClassCount = classCount; // 类别数
  mYoloV5NetWidth = netWidth; // 网络的输入宽
  mYoloV5NetHeight = netHeight; // 网络的输入高
  mMaxOutObject = maxOut; // 最多保留多少个框
  is_segmentation_ = is_segmentation; // 是否是分割模型，涉及到带不带掩码系数？
  mYoloKernel = vYoloKernel; // 应该是每个head的信息，包含 这个head的宽 head的高 anchors等 
  mKernelCount = vYoloKernel.size(); // head数，比如yolov5有3个，每个的尺度是8 16 32。然后每个head有3个anchors，3个头有9个。
