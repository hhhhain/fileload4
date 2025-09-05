/home/ma-user/work/myplugin/MyAddPlugin.cu(19): error: return type is not identical to nor covariant with return type "int32_t" of overridden virtual function "nvinfer1::IPluginV2DynamicExt::enqueue"
      void enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(11): warning #997-D: function "nvinfer1::IPluginV2::getOutputDimensions(int32_t, const nvinfer1::Dims *, int32_t)" is hidden by "AddOnePlugin::getOutputDimensions" -- virtual function override intended?
      DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) {
                ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/home/ma-user/work/myplugin/MyAddPlugin.cu(17): warning #997-D: function "nvinfer1::IPluginV2::getWorkspaceSize(int32_t) const" is hidden by "AddOnePlugin::getWorkspaceSize" -- virtual function override intended?
      size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const { return 0; }
             ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(19): warning #997-D: function "nvinfer1::IPluginV2::enqueue(int32_t, const void *const *, void *const *, void *, cudaStream_t)" is hidden by "AddOnePlugin::enqueue" -- virtual function override intended?
      void enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(34): warning #997-D: function "nvinfer1::IPluginV2Ext::configurePlugin(const nvinfer1::Dims *, int32_t, const nvinfer1::Dims *, int32_t, const nvinfer1::DataType *, const nvinfer1::DataType *, const bool *, const bool *, nvinfer1::PluginFormat, int32_t)" is hidden by "AddOnePlugin::configurePlugin" -- virtual function override intended?
      void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(24): error: object of abstract class type "AddOnePlugin" is not allowed:
            pure virtual function "nvinfer1::IPluginV2DynamicExt::enqueue(const nvinfer1::PluginTensorDesc *, const nvinfer1::PluginTensorDesc *, const void *const *, void *const *, void *, cudaStream_t)" has no overrider
            pure virtual function "nvinfer1::IPluginV2::setPluginNamespace" has no overrider
            pure virtual function "nvinfer1::IPluginV2::getPluginNamespace" has no overrider
      IPluginV2DynamicExt* clone() const { return new AddOnePlugin(); }
                                                      ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(34): error: exception specification for virtual function "AddOnePlugin::configurePlugin" is incompatible with that of overridden function "nvinfer1::IPluginV2DynamicExt::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *, int32_t, const nvinfer1::DynamicPluginTensorDesc *, int32_t)"
      void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(31): error: exception specification for virtual function "AddOnePlugin::getOutputDataType" is incompatible with that of overridden function "nvinfer1::IPluginV2Ext::getOutputDataType"
      DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const {
               ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(28): error: exception specification for virtual function "AddOnePlugin::supportsFormatCombination" is incompatible with that of overridden function "nvinfer1::IPluginV2DynamicExt::supportsFormatCombination"
      bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(27): error: exception specification for virtual function "AddOnePlugin::serialize" is incompatible with that of overridden function "nvinfer1::IPluginV2::serialize"
      void serialize(void* buffer) const {}
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(26): error: exception specification for virtual function "AddOnePlugin::getSerializationSize" is incompatible with that of overridden function "nvinfer1::IPluginV2::getSerializationSize"
      size_t getSerializationSize() const { return 0; }
             ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(25): error: exception specification for virtual function "AddOnePlugin::destroy" is incompatible with that of overridden function "nvinfer1::IPluginV2::destroy"
      void destroy() { delete this; }
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(24): error: exception specification for virtual function "AddOnePlugin::clone" is incompatible with that of overridden function "nvinfer1::IPluginV2DynamicExt::clone"
      IPluginV2DynamicExt* clone() const { return new AddOnePlugin(); }
                           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(24): error: exception specification for virtual function "AddOnePlugin::clone" is incompatible with that of overridden function "nvinfer1::IPluginV2Ext::clone"
      IPluginV2DynamicExt* clone() const { return new AddOnePlugin(); }
                           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(24): error: exception specification for virtual function "AddOnePlugin::clone" is incompatible with that of overridden function "nvinfer1::IPluginV2::clone"
      IPluginV2DynamicExt* clone() const { return new AddOnePlugin(); }
                           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(23): error: exception specification for virtual function "AddOnePlugin::getPluginVersion" is incompatible with that of overridden function "nvinfer1::IPluginV2::getPluginVersion"
      const char* getPluginVersion() const { return "1"; }
                  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(22): error: exception specification for virtual function "AddOnePlugin::getPluginType" is incompatible with that of overridden function "nvinfer1::IPluginV2::getPluginType"
      const char* getPluginType() const { return "AddOnePlugin"; }
                  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(17): error: exception specification for virtual function "AddOnePlugin::getWorkspaceSize" is incompatible with that of overridden function "nvinfer1::IPluginV2DynamicExt::getWorkspaceSize(const nvinfer1::PluginTensorDesc *, int32_t, const nvinfer1::PluginTensorDesc *, int32_t) const"
      size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const { return 0; }
             ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(16): error: exception specification for virtual function "AddOnePlugin::terminate" is incompatible with that of overridden function "nvinfer1::IPluginV2::terminate"
      void terminate() {}
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(15): error: exception specification for virtual function "AddOnePlugin::initialize" is incompatible with that of overridden function "nvinfer1::IPluginV2::initialize"
      int initialize() { return 0; }
          ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(11): error: exception specification for virtual function "AddOnePlugin::getOutputDimensions" is incompatible with that of overridden function "nvinfer1::IPluginV2DynamicExt::getOutputDimensions(int32_t, const nvinfer1::DimsExprs *, int32_t, nvinfer1::IExprBuilder &)"
      DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) {
                ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(10): error: exception specification for virtual function "AddOnePlugin::getNbOutputs" is incompatible with that of overridden function "nvinfer1::IPluginV2::getNbOutputs"
      int getNbOutputs() const { return 2; }
          ^

18 errors detected in the compilation of "/home/ma-user/work/myplugin/MyAddPlugin.cu".
make[2]: *** [CMakeFiles/AddOnePlugin.dir/build.make:76: CMakeFiles/AddOnePlugin.dir/MyAddPlugin.cu.o] Error 2
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/AddOnePlugin.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
