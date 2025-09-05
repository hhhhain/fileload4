// AddOnePlugin.cu
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <vector>
#include <cstring>

using namespace nvinfer1;

// ---------------- CUDA kernel ----------------
__global__ void addOneKernel(const float* input, float* output1, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output1[idx] = input[idx] + 1.0f;
}

// ---------------- Plugin implementation ----------------
class AddOnePlugin : public IPluginV2DynamicExt
{
public:
    AddOnePlugin() {}
    AddOnePlugin(const void* /*data*/, size_t /*length*/) {}
    ~AddOnePlugin() override = default;

    // IPluginV2
    int getNbOutputs() const noexcept override { return 2; } // two outputs

    // IPluginV2DynamicExt
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override
    {
        // outputs have same dims as input
        return inputs[0];
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
    {
        // require float linear for all tensors
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                         const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override
    {
        // nothing to configure for this minimal plugin
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                            const PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    {
        return 0;
    }

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        // compute number of elements in input tensor
        int volume = 1;
        for (int i = 0; i < inputDesc[0].dims.nbDims; ++i) volume *= inputDesc[0].dims.d[i];

        // copy input -> outputs[0] (device to device)
        cudaError_t err = cudaMemcpyAsync(outputs[0], inputs[0], (size_t)volume * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) return -1;

        // launch kernel to compute outputs[1] = input + 1
        const int threads = 256;
        const int blocks = (volume + threads - 1) / threads;
        addOneKernel<<<blocks, threads, 0, stream>>>((const float*)inputs[0], (float*)outputs[1], volume);
        // optional: check kernel launch error (can't return cudaError_t directly)
        err = cudaGetLastError();
        if (err != cudaSuccess) return -2;

        return 0;
    }

    // IPluginV2Ext / IPluginV2DynamicExt required methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        return inputTypes[0];
    }

    // lifecycle
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}

    // serialization (none)
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void* /*buffer*/) const noexcept override {}

    void destroy() noexcept override { delete this; }

    IPluginV2DynamicExt* clone() const noexcept override { return new AddOnePlugin(); }

    // namespace
    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace ? libNamespace : ""; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    // plugin identity
    const char* getPluginType() const noexcept override { return "AddOnePlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }

private:
    std::string mNamespace;
};

// ---------------- Plugin Creator ----------------
class AddOnePluginCreator : public IPluginCreator
{
public:
    AddOnePluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }

    const char* getPluginName() const noexcept override { return "AddOnePlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(const char* /*name*/, const PluginFieldCollection* /*fc*/) noexcept override
    {
        return new AddOnePlugin();
    }

    IPluginV2* deserializePlugin(const char* /*name*/, const void* serialData, size_t serialLength) noexcept override
    {
        return new AddOnePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace ? libNamespace : ""; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    PluginFieldCollection mFC{};
};

// register plugin creator
REGISTER_TENSORRT_PLUGIN(AddOnePluginCreator);
