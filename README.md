#include "NvInfer.h"
#include "cuda_runtime.h"
#include <cassert>

using namespace nvinfer1;

__global__ void add_one_kernel(const float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + 1.0f;
    }
}

class AddOnePlugin : public IPluginV2DynamicExt
{
public:
    AddOnePlugin() {}
    AddOnePlugin(const void* data, size_t length) {}

    int getNbOutputs() const noexcept override { return 1; }
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        return inputs[0]; // 输出尺寸 = 输入尺寸
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
    {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    {
        return 0;
    }

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        int size = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
            size *= inputDesc[0].dims.d[i];

        add_one_kernel<<<(size + 255)/256, 256, 0, stream>>>((const float*)inputs[0], (float*)outputs[0], size);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* buffer) const noexcept override {}
    const char* getPluginType() const noexcept override { return "AddOnePlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2DynamicExt* clone() const noexcept override { return new AddOnePlugin(); }

    void setPluginNamespace(const char* libNamespace) noexcept override {}
    const char* getPluginNamespace() const noexcept override { return ""; }

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        return DataType::kFLOAT;
    }
};
