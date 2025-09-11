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

  CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*))); // 这句话是在gpu分配内存的。比如mKernelCount为3，有3个头，分配3块内存区域，分配存3个头。
  size_t AnchorLen = sizeof(float)* kNumAnchor * 2; // 没看到kNumAnchor的定义，应该是每个head的anchor数，比如yolov5的anchor数是3，*2是因为有长宽两个属性。
  for (int ii = 0; ii < mKernelCount; ii++) { // 遍历，每次处理一个head。
    CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen)); // 这一步是进一步给每个头具体分配3个anchor的地址大小。把GPU分配的内存地址写入mAnchor
    const auto& yolo = mYoloKernel[ii];
    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice)); // 分别是gpu 内存地址，cpu内存里的数据比如([10,13],[16,30],[33,23])，拷贝的字节数，（拷贝方向）从cpu到gpu。这句话进一步分配宽 高等内存。
  }
}

YoloLayerPlugin::~YoloLayerPlugin() {  // 这是析构函数，前面有个波浪号~，对象结束的时候自动调用执行。
  for (int ii = 0; ii < mKernelCount; ii++) {
    CUDA_CHECK(cudaFree(mAnchor[ii]));
  }
  CUDA_CHECK(cudaFreeHost(mAnchor));
}

// create the plugin at runtime from a byte stream
// tensorrt的plugin有两种用法：1.直接创建，如上面那个函数。2.从序列化数据恢复，在engine load时执行，用这个构造函数。这里是构造函数重载实现，调用时会根据传入的参数不同选择不同的实现。
// data是原始字节流，length时字节流的总字节数。从这个data里面恢复。
// 最终获得了plugin的成员变量。以及GPU里面存入了：
// mAnchor[0] > 指向GPU内存，里面是 ([10,13],[16,30],[33,23])
// mAnchor[1] > 指向GPU内存，里面是 ([30,61],[62,45],[59,119])
// mAnchor[2] > 指向GPU内存，里面是 ([116,90],[156,198],[373,326])
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
  using namespace Tn; // 使用命名空间Tn下的函数，write和read应该就是。
  const char *d = reinterpret_cast<const char *>(data), *a = d; // 把void传入的data转换类型，命名为d，同时把a设为d的拷贝。
  read(d, mClassCount); // 从d开始读，并移动指针，所以下面都是从d开始读的，因为指针跟着变了。里面的内容是创建的时候写入的。这些都是plugin的成员变量，会在后面的时候使用。访问成员变量来用。
  read(d, mThreadCount);
  read(d, mKernelCount);
  read(d, mYoloV5NetWidth);
  read(d, mYoloV5NetHeight);
  read(d, mMaxOutObject);
  read(d, is_segmentation_);
  mYoloKernel.resize(mKernelCount); // 调整容器大小，比如有3个头。
  auto kernelSize = mKernelCount * sizeof(YoloKernel); // 头数乘以大小。
  memcpy(mYoloKernel.data(), d, kernelSize); // 继续从d中读取。不是read函数，不自动加指针。所以下一条手动d+指针。
  d += kernelSize;
  // for (int i = 0; i < mKernelCount; ++i)
  //     read(d, mYoloKernel[i]);  
  
  CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*))); // 跟上面一样，这句话是在gpu分配内存的。比如mKernelCount为3，有3个头，分配3块内存区域，分配存3个头。
  size_t AnchorLen = sizeof(float)* kNumAnchor * 2; // 跟上面一样，没看到kNumAnchor的定义，应该是每个head的anchor数，比如yolov5的anchor数是3，*2是因为有长宽两个属性。
  for (int ii = 0; ii < mKernelCount; ii++) { // 跟上面一样，遍历，每次处理一个head。 
    CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen)); // 跟上面一样，这一步是进一步给每个头具体分配3个anchor的地址大小。把GPU分配的内存地址写入mAnchor
    const auto& yolo = mYoloKernel[ii];
    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice)); // 跟上面一样，分别是gpu内存地址，cpu内存里的数据比如([10,13],[16,30],[33,23])，拷贝的字节数，（拷贝方向）从cpu到gpu。这句话进一步分配宽 高等内存。
  }
  assert(d == a + length); // 起个判断作用，按道理说d现在的指针应该要等于所有的信息量总长度，如果不是的话说明中间出了问题。



  // printf("mThreadCount is %d\n", mThreadCount);    
  // printf("mMaxOutObject is %d\n", mMaxOutObject);  
  // printf("mKernelCount is %d\n", mKernelCount);
  // printf("mYoloV5NetWidth is %d\n", mYoloV5NetWidth);  
  // printf("mYoloV5NetHeight is %d\n", mYoloV5NetHeight);   
  // printf("mClassCount is %d\n", mClassCount);    




}

// 序列化，把关键超参数比如类别数、网络尺寸、每个head的kernel信息写入字节流，以便保存成engine。这个buffer就是地址，写东西进去的地址。engine会把这个内存里的东西保存下来，以便恢复。
// 应该是从上面的YoloLayerPlugin重载实现来恢复的。
void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {  
  using namespace Tn;
  char* d = static_cast<char*>(buffer), *a = d;
  write(d, mClassCount);
  write(d, mThreadCount);
  write(d, mKernelCount); // 比如mKernelCount为3，有3个头    1
  write(d, mYoloV5NetWidth);
  write(d, mYoloV5NetHeight);
  write(d, mMaxOutObject);
  write(d, is_segmentation_);
  auto kernelSize = mKernelCount * sizeof(YoloKernel);
  memcpy(d, mYoloKernel.data(), kernelSize);
  d += kernelSize; // 同理。
  // for (int i = 0; i < mKernelCount; ++i)
  //     write(d, mYoloKernel[i]);  
  

  assert(d == a + getSerializationSize()); // 同理，用下面的函数计算，是否相等。


  // printf("mThreadCount is %d\n", mThreadCount);    
  // printf("mMaxOutObject is %d\n", mMaxOutObject);  
  // printf("mKernelCount is %d\n", mKernelCount);
  // printf("mYoloV5NetWidth is %d\n", mYoloV5NetWidth);  
  // printf("mYoloV5NetHeight is %d\n", mYoloV5NetHeight);   
  // printf("mClassCount is %d\n", mClassCount);     





}

// 计算serialize()将写入的总字节数，二者必须严格一致。
size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
  size_t s = sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount);
  s += sizeof(YoloKernel) * mYoloKernel.size();
  s += sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight);
  s += sizeof(mMaxOutObject) + sizeof(is_segmentation_);
  return s;
}

// 初始化
int YoloLayerPlugin::initialize() TRT_NOEXCEPT {
  // printf("initial here\n");
  return 0;
}

// 
Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT {
  // output the result to channel
  // 为什么传入totalsize + 1, 1, 1还未知，求出每个检测的float数量，乘以最大数量，得到总size。
  // Detection 是结构体{float bbox[4], float conf, float class_id, float mask[32];}
  // mMaxOutObject是前面提到的plugin成员变量。

  int total_size = mMaxOutObject * sizeof(Detection) / sizeof(float);
  // 打印调试信息
  // std::cout << "[YoloLayerPlugin::getOutputDimensions] index=" << index
  //           << " nbInputDims=" << nbInputDims
  //           << " total_size=" << total_size
  //           << " -> return Dims3(" << (total_size + 1) << ", 1, 1)" 
  //           << std::endl;  
  return Dims3(total_size + 1, 1, 1);
}

// Set plugin namespace
// 用来区分不同的版本，但没看到有哪些版本？上面只是函数重载，不属于区分不同的版本。
void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT {
  mPluginNamespace = pluginNamespace;
}

// 获得当前查询的这个命名空间
const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
  return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
// 固定返回kfloat
DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT {
  return false;
}

void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT {}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

// Detach the plugin object from its execution context.
void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT {
  return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

// 由tensorrt不再需要plugin实例时调用，插件负责自我销毁。
void YoloLayerPlugin::destroy() TRT_NOEXCEPT {
  delete this;
}

// Clone the plugin
// 深拷贝。
IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT {
  YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, is_segmentation_, mYoloKernel);
  p->setPluginNamespace(mPluginNamespace);
  return p;
}

// __device__是cuda的函数修饰符，只能被GPU上的其他函数或者kernel调用。
// 这是sigmoid激活函数。输出在[0, 1]之间。 bonding box和置信度通常需要这个。
__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

// __global__是在GPU运行，但可以从CPU调用。__host__是在CPU运行。
// CalDetection是核心kernel。把网络的输出feature map转换为Detection结构体，也就是bbox+conf+class_id，并写入输出buffer。
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

  // printf("111111111111\n");
  // printf("noElements is %d\n", noElements);
  // return;
  if (idx >= noElements) return; 
      // printf("222222222222\n");
  int total_grid = yoloWidth * yoloHeight;
  // printf("3333333333333\n");
  // printf("total_grid is %d\n", total_grid);
  int bnIdx = idx / total_grid; // batch的索引，相当于第几张图，10张图的第几张。
  // printf("bnIdx:  %d\n", bnIdx);
  idx = idx - total_grid * bnIdx;// 是偏移。相对于当前batch图像的起始位置的网格索引。
  int info_len_i = 5 + classes; //每个anchor的元素数量，这个好理解，比如5+80.5是由boss，conf组成。
  if (is_segmentation) info_len_i += 32; //分割。
  // printf("4444444444444\n");
  // kNumAnchor如前文猜测应该是每个head的anchor数，乘以格子数，乘以每个格子的信息量，再乘以第几张图，然后对input指针进行偏移，指向某一张图的起始位置。
  const float* curInput = input + bnIdx * (info_len_i * total_grid * kNumAnchor); 
  // printf("55555555555555\n");
  // kNumAnchor如前文猜测应该是每个head的anchor数，每个cell同样，比如是3. 遍历每个anchor， 
  for (int k = 0; k < kNumAnchor; ++k) {
    // 这条语句是在对某一张图的所有cell求置信度，0或者1. 为什么是所有cell？因为每一个idx都绑定了线程，都是并行执行的。[anchor1][anchor2][anchor3]的顺序在内存中存放。
    float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
    // printf("666666666666666\n");
    // 所有cell都是并行的，所以如果判断出当前的cell的置信度太低的话，就不管这个cell了。执行下一次for，也就是下一个anchor，同理。
    if (box_prob < kIgnoreThresh) continue;
    // printf("7777777777777777\n");
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
    
      // printf("bnIdx:  %d\n", bnIdx);
    // printf("res_count address: %p, value: %f\n", res_count, *res_count);
    int count = (int)atomicAdd(res_count, 1);
    // printf("8888888888888888\n");
    if (count >= maxoutobject) return;
    // printf("99999999999999999999\n");
    // 每个cell计算自己应该要写的位置，转成detection指针，以后要写的时候就可以用det->结构。比如下面。
    char *data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
    // printf("10101010101010101\n");
    Detection *det = (Detection*)(data);
    // printf("121212121212121212\n");
    // if (bnIdx==0){
    // printf("bnIdx:  %d\n", bnIdx);
    // printf("res_count address: %p, value: %f, atomic count: %d\n", res_count, *res_count, count);}
    // return;
    // if (tid < 3) {
    //     printf("Thread %d: res_count address: %p, value: %f, atomic count: %d\n", tid, res_count, *res_count, count);
    // }
    // printf("13131313131313131313131\n");
    // 如果 tid >=3 不想执行后续逻辑，就直接 return
    // if (tid >= 3) return;
    // printf("14141414141414141414\n");
    // 定位行和列。   
    // printf("yoloWidth is %d\n, yoloHeight is %d\n", yoloWidth, yoloHeight);  
    int row = idx / yoloWidth;
    int col = idx % yoloWidth;
    // printf("1515151515151515\n");
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

void YoloLayerPlugin::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize) {
  // if(mThreadCount==0) mThreadCount = 256;
  // printf("1111mThreadCount is %d\n", mThreadCount);
  int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float); // 这条单张输出的大小，有个mMaxOutObject的限制，限制了最多输出多少个框。
  // printf("2222mThreadCount is %d\n", mThreadCount);
  for (int idx = 0; idx < batchSize; ++idx) {
    // 注意看这里的idx在for循环里已经是bs的索引了，目的是把整个bs里的每张图的框计数res_count清零，表示还没写入任何框。
    // printf("cudaMemsetAsync here\n");
    // printf("3333mThreadCount is %d\n", mThreadCount);

    CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
    // printf("4444mThreadCount is %d\n", mThreadCount);

  }
  int numElem = 0;
  // printf("5555mThreadCount is %d\n", mThreadCount);

  // printf("11 here\n");
  for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
    // printf("6666mThreadCount is %d\n", mThreadCount);

    // printf("22 here\n");
    const auto& yolo = mYoloKernel[i];
    // printf("7mThreadCount is %d\n", mThreadCount);

    // printf("33 here\n");
    // grid数乘以bs
    numElem = yolo.width * yolo.height * batchSize;
    // printf("8mThreadCount is %d\n", mThreadCount);
    // printf("44 here\n");
    // 每个cell对应一个线程，如果线程大于总cell数了，那就降低线程的数量。而mThreadCount一般都比numElem小得多。
    if (numElem < mThreadCount) mThreadCount = numElem;
    // printf("9mThreadCount is %d\n", mThreadCount);
    // printf("55 here\n");
    // 这里的用法是cuda kernel的调用。假设用__global__定义了一个函数fun。那么执行的时候就必须这样调用执行 fun<<<(四个内核参数)>>>(自己定义的输入参数)
    // 这里的内核参数是控制线程数的，分别是<<<gridsize, blocksize, sharedmem, stream>>>。
    // 先看第二个参数，blocksize决定了每个block能处理多少个cell，这里设置为mThreadCount，已知一般情况下mThreadCount都比numElem小得多，需要计算到底需要多少个mThreadCount
    // 第一个参数是向上取整的写法。计算出需要多少个block。这样保证了每个cell都有一个线程进行处理。
    // printf("mThreadCount is %d\n", mThreadCount);  
    // printf("mMaxOutObject is %d\n", mMaxOutObject);
    // printf("numElem is %d\n", numElem);
    // printf("mYoloV5NetWidth is %d\n", mYoloV5NetWidth);
    // printf("mYoloV5NetHeight is %d\n", mYoloV5NetHeight);
    // printf("yolo.width is %d\n", yolo.width);
    // printf("yolo.height is %d\n", yolo.height);
    // printf("mClassCount is %d\n", mClassCount);
    // printf("outputElem is %d\n", outputElem);
    CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
      (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem, is_segmentation_);
  }
}


// tensorrt执行推理时调用。写入output。
int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  // printf("enqueue here\n");
  forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
  return 0;
}

// 定义一个插件，mPluginAttributes存放哥哥字段，比如名字类型指针。mfc指向mPluginAttributes，下面的mFC.fields = mPluginAttributes.data()。
PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

YoloPluginCreator::YoloPluginCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const TRT_NOEXCEPT {
  return "YoloLayer_TRT";
}

const char* YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

// tensorrt通过mfc就能得到插件的所有参数信息
const PluginFieldCollection* YoloPluginCreator::getFieldNames() TRT_NOEXCEPT {
  return &mFC;
}

// 创建一个新的yolo plugin实例，返回值是IPluginV2IOExt*，表示可以插入到网络中的自定义层。
// 具体实例化在model.cpp文件中.
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


  // std::cout << "max_output_object_count=" << max_output_object_count
  //         << std::endl;


  // 接下来解析kernels字段。
  // 使用memcpy拿到这个字段的内容，描述了输出特征图的尺寸和anchor信息。
  std::vector<YoloKernel> kernels(fc->fields[1].length);
  memcpy(&kernels[0], fc->fields[1].data, kernels.size() * sizeof(YoloKernel));

  // 利用这两个字段解析出的内容创建具体plugin实例。
  YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, is_segmentation, kernels);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

// 从engine中序列化一个plugin，YoloLayerPlugin会用到上面的函数重载来实现。
IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call YoloLayerPlugin::destroy()
  // printf("running deserializePlugin\n");
  YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}
}





















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

  CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*))); // 这句话是在gpu分配内存的。比如mKernelCount为3，有3个头，分配3块内存区域，分配存3个头。
  size_t AnchorLen = sizeof(float)* kNumAnchor * 2; // 没看到kNumAnchor的定义，应该是每个head的anchor数，比如yolov5的anchor数是3，*2是因为有长宽两个属性。
  for (int ii = 0; ii < mKernelCount; ii++) { // 遍历，每次处理一个head。
    CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen)); // 这一步是进一步给每个头具体分配3个anchor的地址大小。把GPU分配的内存地址写入mAnchor
    const auto& yolo = mYoloKernel[ii];
    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice)); // 分别是gpu 内存地址，cpu内存里的数据比如([10,13],[16,30],[33,23])，拷贝的字节数，（拷贝方向）从cpu到gpu。这句话进一步分配宽 高等内存。
  }
}

YoloLayerPlugin::~YoloLayerPlugin() {  // 这是析构函数，前面有个波浪号~，对象结束的时候自动调用执行。
  for (int ii = 0; ii < mKernelCount; ii++) {
    CUDA_CHECK(cudaFree(mAnchor[ii]));
  }
  CUDA_CHECK(cudaFreeHost(mAnchor));
}

// create the plugin at runtime from a byte stream
// tensorrt的plugin有两种用法：1.直接创建，如上面那个函数。2.从序列化数据恢复，在engine load时执行，用这个构造函数。这里是构造函数重载实现，调用时会根据传入的参数不同选择不同的实现。
// data是原始字节流，length时字节流的总字节数。从这个data里面恢复。
// 最终获得了plugin的成员变量。以及GPU里面存入了：
// mAnchor[0] > 指向GPU内存，里面是 ([10,13],[16,30],[33,23])
// mAnchor[1] > 指向GPU内存，里面是 ([30,61],[62,45],[59,119])
// mAnchor[2] > 指向GPU内存，里面是 ([116,90],[156,198],[373,326])
YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length) {
  using namespace Tn; // 使用命名空间Tn下的函数，write和read应该就是。
  const char *d = reinterpret_cast<const char *>(data), *a = d; // 把void传入的data转换类型，命名为d，同时把a设为d的拷贝。
  read(d, mClassCount); // 从d开始读，并移动指针，所以下面都是从d开始读的，因为指针跟着变了。里面的内容是创建的时候写入的。这些都是plugin的成员变量，会在后面的时候使用。访问成员变量来用。
  read(d, mThreadCount);
  read(d, mKernelCount);
  read(d, mYoloV5NetWidth);
  read(d, mYoloV5NetHeight);
  read(d, mMaxOutObject);
  read(d, is_segmentation_);
  mYoloKernel.resize(mKernelCount); // 调整容器大小，比如有3个头。
  auto kernelSize = mKernelCount * sizeof(YoloKernel); // 头数乘以大小。
  memcpy(mYoloKernel.data(), d, kernelSize); // 继续从d中读取。不是read函数，不自动加指针。所以下一条手动d+指针。
  d += kernelSize;
  CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*))); // 跟上面一样，这句话是在gpu分配内存的。比如mKernelCount为3，有3个头，分配3块内存区域，分配存3个头。
  size_t AnchorLen = sizeof(float)* kNumAnchor * 2; // 跟上面一样，没看到kNumAnchor的定义，应该是每个head的anchor数，比如yolov5的anchor数是3，*2是因为有长宽两个属性。
  for (int ii = 0; ii < mKernelCount; ii++) { // 跟上面一样，遍历，每次处理一个head。 
    CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen)); // 跟上面一样，这一步是进一步给每个头具体分配3个anchor的地址大小。把GPU分配的内存地址写入mAnchor
    const auto& yolo = mYoloKernel[ii];
    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice)); // 跟上面一样，分别是gpu内存地址，cpu内存里的数据比如([10,13],[16,30],[33,23])，拷贝的字节数，（拷贝方向）从cpu到gpu。这句话进一步分配宽 高等内存。
  }
  assert(d == a + length); // 起个判断作用，按道理说d现在的指针应该要等于所有的信息量总长度，如果不是的话说明中间出了问题。
}

// 序列化，把关键超参数比如类别数、网络尺寸、每个head的kernel信息写入字节流，以便保存成engine。这个buffer就是地址，写东西进去的地址。engine会把这个内存里的东西保存下来，以便恢复。
// 应该是从上面的YoloLayerPlugin重载实现来恢复的。
void YoloLayerPlugin::serialize(void* buffer) const TRT_NOEXCEPT {  
  using namespace Tn;
  char* d = static_cast<char*>(buffer), *a = d;
  write(d, mClassCount);
  write(d, mThreadCount);
  write(d, mKernelCount); // 比如mKernelCount为3，有3个头
  write(d, mYoloV5NetWidth);
  write(d, mYoloV5NetHeight);
  write(d, mMaxOutObject);
  write(d, is_segmentation_);
  auto kernelSize = mKernelCount * sizeof(YoloKernel);
  memcpy(d, mYoloKernel.data(), kernelSize);
  d += kernelSize; // 同理。

  assert(d == a + getSerializationSize()); // 同理，用下面的函数计算，是否相等。
}

// 计算serialize()将写入的总字节数，二者必须严格一致。
size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT {
  size_t s = sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount);
  s += sizeof(YoloKernel) * mYoloKernel.size();
  s += sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight);
  s += sizeof(mMaxOutObject) + sizeof(is_segmentation_);
  return s;
}

// 初始化
int YoloLayerPlugin::initialize() TRT_NOEXCEPT {
  return 0;
}

// 
Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT {
  // output the result to channel
  // 为什么传入totalsize + 1, 1, 1还未知，求出每个检测的float数量，乘以最大数量，得到总size。
  // Detection 是结构体{float bbox[4], float conf, float class_id, float mask[32];}
  // mMaxOutObject是前面提到的plugin成员变量。
  int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);
  return Dims3(totalsize + 1, 1, 1);
}

// Set plugin namespace
// 用来区分不同的版本，但没看到有哪些版本？上面只是函数重载，不属于区分不同的版本。
void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT {
  mPluginNamespace = pluginNamespace;
}

// 获得当前查询的这个命名空间
const char* YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT {
  return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
// 固定返回kfloat
DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT {
  return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT {
  return false;
}

void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT {}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {}

// Detach the plugin object from its execution context.
void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

const char* YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT {
  return "YoloLayer_TRT";
}

const char* YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

// 由tensorrt不再需要plugin实例时调用，插件负责自我销毁。
void YoloLayerPlugin::destroy() TRT_NOEXCEPT {
  delete this;
}

// Clone the plugin
// 深拷贝。
IPluginV2IOExt* YoloLayerPlugin::clone() const TRT_NOEXCEPT {
  YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, is_segmentation_, mYoloKernel);
  p->setPluginNamespace(mPluginNamespace);
  return p;
}

// __device__是cuda的函数修饰符，只能被GPU上的其他函数或者kernel调用。
// 这是sigmoid激活函数。输出在[0, 1]之间。 bonding box和置信度通常需要这个。
__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

// __global__是在GPU运行，但可以从CPU调用。__host__是在CPU运行。
// CalDetection是核心kernel。把网络的输出feature map转换为Detection结构体，也就是bbox+conf+class_id，并写入输出buffer。
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

void YoloLayerPlugin::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize) {
  int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float); // 这条单张输出的大小，有个mMaxOutObject的限制，限制了最多输出多少个框。
  for (int idx = 0; idx < batchSize; ++idx) {
    // 注意看这里的idx在for循环里已经是bs的索引了，目的是把整个bs里的每张图的框计数res_count清零，表示还没写入任何框。
    CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
  }
  int numElem = 0;
  for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
    const auto& yolo = mYoloKernel[i];
    // grid数乘以bs
    numElem = yolo.width * yolo.height * batchSize;
    // 每个cell对应一个线程，如果线程大于总cell数了，那就降低线程的数量。而mThreadCount一般都比numElem小得多。
    if (numElem < mThreadCount) mThreadCount = numElem;
    
    // 这里的用法是cuda kernel的调用。假设用__global__定义了一个函数fun。那么执行的时候就必须这样调用执行 fun<<<(四个内核参数)>>>(自己定义的输入参数)
    // 这里的内核参数是控制线程数的，分别是<<<gridsize, blocksize, sharedmem, stream>>>。
    // 先看第二个参数，blocksize决定了每个block能处理多少个cell，这里设置为mThreadCount，已知一般情况下mThreadCount都比numElem小得多，需要计算到底需要多少个mThreadCount
    // 第一个参数是向上取整的写法。计算出需要多少个block。这样保证了每个cell都有一个线程进行处理。
    CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
      (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem, is_segmentation_);
  }
}


// tensorrt执行推理时调用。写入output。
int YoloLayerPlugin::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT {
  forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
  return 0;
}

// 定义一个插件，mPluginAttributes存放哥哥字段，比如名字类型指针。mfc指向mPluginAttributes，下面的mFC.fields = mPluginAttributes.data()。
PluginFieldCollection YoloPluginCreator::mFC{};
std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

YoloPluginCreator::YoloPluginCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator::getPluginName() const TRT_NOEXCEPT {
  return "YoloLayer_TRT";
}

const char* YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

// tensorrt通过mfc就能得到插件的所有参数信息
const PluginFieldCollection* YoloPluginCreator::getFieldNames() TRT_NOEXCEPT {
  return &mFC;
}

// 创建一个新的yolo plugin实例，返回值是IPluginV2IOExt*，表示可以插入到网络中的自定义层。
// 具体实例化在model.cpp文件中.
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

  // 接下来解析kernels字段。
  // 使用memcpy拿到这个字段的内容，描述了输出特征图的尺寸和anchor信息。
  std::vector<YoloKernel> kernels(fc->fields[1].length);
  memcpy(&kernels[0], fc->fields[1].data, kernels.size() * sizeof(YoloKernel));

  // 利用这两个字段解析出的内容创建具体plugin实例。
  YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, is_segmentation, kernels);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

// 从engine中序列化一个plugin，YoloLayerPlugin会用到上面的函数重载来实现。
IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call YoloLayerPlugin::destroy()
  YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}
}

