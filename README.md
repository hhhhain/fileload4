我打印network.num_layers得到：
  output[0] shape: (10, 3, 20, 34, 2)
index=286, name=/model.24/Constant_21_output_0, type=LayerType.CONSTANT, nb_outputs=1
  output[0] shape: ()
index=287, name=(Unnamed Layer* 287) [Shuffle], type=LayerType.SHUFFLE, nb_outputs=1
  output[0] shape: (1, 1, 1, 1, 1)
index=288, name=/model.24/Pow_2, type=LayerType.ELEMENTWISE, nb_outputs=1
  output[0] shape: (10, 3, 20, 34, 2)
index=289, name=/model.24/Constant_22_output_0, type=LayerType.CONSTANT, nb_outputs=1
  output[0] shape: (1, 3, 20, 34, 2)
index=290, name=/model.24/Mul_11, type=LayerType.ELEMENTWISE, nb_outputs=1
  output[0] shape: (10, 3, 20, 34, 2)
index=291, name=/model.24/Concat_2, type=LayerType.CONCATENATION, nb_outputs=1
  output[0] shape: (10, 3, 20, 34, 31)
index=292, name=/model.24/Reshape_5, type=LayerType.SHUFFLE, nb_outputs=1
  output[0] shape: (10, 2040, 31)
index=293, name=/model.24/Concat_3, type=LayerType.CONCATENATION, nb_outputs=1
  output[0] shape: (10, 42840, 31)
我搜索没发现有93的结果，但是打印model，发现：
    )
    (24): Detect(
      (m): ModuleList(
        (0): Conv2d(128, 93, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(256, 93, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(512, 93, kernel_size=(1, 1), stride=(1, 1))
      )
    )却包含有93的层，怎么回事？
