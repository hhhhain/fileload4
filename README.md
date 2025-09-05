cmake_minimum_required(VERSION 3.18)
project(AddOnePlugin LANGUAGES CXX CUDA)

# TensorRT 安装路径
set(TENSORRT_ROOT /usr/local/TensorRT)  # 修改为你实际路径

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# CUDA 编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含目录
include_directories(
    ${TENSORRT_ROOT}/include
    /usr/local/cuda/include
)

# 库目录
link_directories(
    ${TENSORRT_ROOT}/lib
)

# 编译动态库
add_library(AddOnePlugin SHARED AddOnePlugin.cu)

# 链接 TensorRT 和 CUDA 库
target_link_libraries(AddOnePlugin
    nvinfer
    nvinfer_plugin
    cudart
)





auto creator = getPluginRegistry()->getPluginCreator("AddOnePlugin", "1");
PluginFieldCollection fc{};
IPluginV2* plugin = creator->createPlugin("addone", &fc);

// 假设 input_tensor 是 engine 原始输出
ITensor* inputTensors[] = { input_tensor };
auto addone_layer = network->addPluginV2(inputTensors, 1, *plugin);

// 输出两个 tensor
addone_layer->getOutput(0)->setName("output0");  // 原始
addone_layer->getOutput(1)->setName("output1");  // +1
network->markOutput(*addone_layer->getOutput(0));
network->markOutput(*addone_layer->getOutput(1));
