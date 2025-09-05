/home/ma-user/work/myplugin/MyAddPlugin.cu(5): error: expected an identifier
  class AddOnePlugin : public IPluginV2DynamicExt {
  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(6): error: expected an expression
  public:
  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(22): warning #12-D: parsing restarts here after previous syntax error
                   const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
                                                                                                                           ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

/home/ma-user/work/myplugin/MyAddPlugin.cu(22): error: expected a "}"
                   const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
                                                                                                                           ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(24): error: a type qualifier is not allowed on a nonmember function
      const char* getPluginType() const noexcept override { return "AddOnePlugin"; }
                                  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(24): error: expected a "{"
      const char* getPluginType() const noexcept override { return "AddOnePlugin"; }
                                                 ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(25): error: a type qualifier is not allowed on a nonmember function
      const char* getPluginVersion() const noexcept override { return "1"; }
                                     ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(25): error: expected a "{"
      const char* getPluginVersion() const noexcept override { return "1"; }
                                                    ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(26): error: a type qualifier is not allowed on a nonmember function
      IPluginV2DynamicExt* clone() const noexcept override { return new AddOnePlugin(); }
                                   ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(26): error: expected a "{"
      IPluginV2DynamicExt* clone() const noexcept override { return new AddOnePlugin(); }
                                                  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(26): error: expected a type specifier
      IPluginV2DynamicExt* clone() const noexcept override { return new AddOnePlugin(); }
                                                                        ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(27): error: expected a "{"
      void destroy() noexcept override { delete this; }
                              ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(27): error: "this" may only be used inside a nonstatic member function
      void destroy() noexcept override { delete this; }
                                                ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(28): error: a type qualifier is not allowed on a nonmember function
      size_t getSerializationSize() const noexcept override { return 0; }
                                    ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(28): error: expected a "{"
      size_t getSerializationSize() const noexcept override { return 0; }
                                                   ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(29): error: a type qualifier is not allowed on a nonmember function
      void serialize(void* buffer) const noexcept override {}
                                   ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(29): error: expected a "{"
      void serialize(void* buffer) const noexcept override {}
                                                  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(30): error: expected a "{"
      bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
                                                                                                                   ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(33): error: a type qualifier is not allowed on a nonmember function
      DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
                                                                                      ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(33): error: expected a "{"
      DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
                                                                                                     ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(37): error: expected a "{"
                           const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}
                                                                                       ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(38): error: expected a declaration
  };
  ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(48): error: incomplete type is not allowed
  void AddOnePlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
       ^

/home/ma-user/work/myplugin/MyAddPlugin.cu(48): error: expected a ";"
  void AddOnePlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                   ^

23 errors detected in the compilation of "/home/ma-user/work/myplugin/MyAddPlugin.cu".
make[2]: *** [CMakeFiles/AddOnePlugin.dir/build.make:76: CMakeFiles/AddOnePlugin.dir/MyAddPlugin.cu.o] Error 2
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/AddOnePlugin.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
