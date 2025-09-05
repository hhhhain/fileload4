[ 50%] Building CUDA object CMakeFiles/AddOnePlugin.dir/MyAddPlugin.cu.o
/home/ma-user/work/myplugin/MyAddPlugin.cu(103): error: namespace "std" has no member "string"
      std::string mNamespace;
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(29): warning #997-D: function "nvinfer1::IPluginV2::getOutputDimensions(int32_t, const nvinfer1::Dims *, int32_t)" is hidden by "AddOnePlugin::getOutputDimensions" -- virtual function override intended?
      DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/home/ma-user/work/myplugin/MyAddPlugin.cu(48): warning #997-D: function "nvinfer1::IPluginV2::getWorkspaceSize(int32_t) const" is hidden by "AddOnePlugin::getWorkspaceSize" -- virtual function override intended?
      size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
             ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(54): warning #997-D: function "nvinfer1::IPluginV2::enqueue(int32_t, const void *const *, void *const *, void *, cudaStream_t)" is hidden by "AddOnePlugin::enqueue" -- virtual function override intended?
      int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
          ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(42): warning #997-D: function "nvinfer1::IPluginV2Ext::configurePlugin(const nvinfer1::Dims *, int32_t, const nvinfer1::Dims *, int32_t, const nvinfer1::DataType *, const nvinfer1::DataType *, const bool *, const bool *, nvinfer1::PluginFormat, int32_t)" is hidden by "AddOnePlugin::configurePlugin" -- virtual function override intended?
      void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(134): error: namespace "std" has no member "string"
      std::string mNamespace;
           ^

2 errors detected in the compilation of "/home/ma-user/work/myplugin/MyAddPlugin.cu".
make[2]: *** [CMakeFiles/AddOnePlugin.dir/build.make:76: CMakeFiles/AddOnePlugin.dir/MyAddPlugin.cu.o] Error 2
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/AddOnePlugin.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
