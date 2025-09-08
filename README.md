cmake_minimum_required(VERSION 3.18)
project(post_process_before_nms LANGUAGES CXX CUDA)

# TensorRT 安装路径
set(TENSORRT_ROOT /home/ma-user/work/copy/files/TensorRT-8.6.1.6)  # 修改为你实际路径

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# CUDA 编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含目录
include_directories(
    ${TENSORRT_ROOT}/include
    /usr/local/cuda-12.1/include
)

# 库目录
link_directories(
    ${TENSORRT_ROOT}/lib
)

# 编译动态库
add_library(post_process_before_nms SHARED yololayer.cu)

# 链接 TensorRT 和 CUDA 库
target_link_libraries(post_process_before_nms
    nvinfer
    nvinfer_plugin
    cudart
)
