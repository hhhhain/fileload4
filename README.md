#include <cuda_runtime.h>
#include <NvInfer.h>
using namespace nvinfer1;

class AddOnePlugin : public IPluginV2DynamicExt {
public:
    AddOnePlugin() {}
    AddOnePlugin(const void* data, size_t length) {}
    
    int getNbOutputs() const noexcept override { return 2; }  // 原始 + 加1
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override {
        return inputs[0];  // 输出 shape = 输入 shape
    }
    
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override { return 0; }
    
    void enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                 const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    
    const char* getPluginType() const noexcept override { return "AddOnePlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    IPluginV2DynamicExt* clone() const noexcept override { return new AddOnePlugin(); }
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override {}
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
        return inputTypes[0];
    }
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                         const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}
};

// CUDA kernel
__global__ void addOneKernel(const float* input, float* output1, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output1[idx] = input[idx] + 1.0f;
    }
}

void AddOnePlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                           const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int volume = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
        volume *= inputDesc[0].dims.d[i];

    // output0 = 原始 input
    cudaMemcpyAsync(outputs[0], inputs[0], volume * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // output1 = input + 1
    int threads = 256;
    int blocks = (volume + threads - 1) / threads;
    addOneKernel<<<blocks, threads, 0, stream>>>((const float*)inputs[0], (float*)outputs[1], volume);
}
