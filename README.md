det1 = network.get_layer(242).get_output(0)
det1_fp32 = network.add_identity(det1).get_output(0)
det1_fp32.dtype = trt.DataType.FLOAT

det2 = network.get_layer(267).get_output(0)
det2_fp32 = network.add_identity(det2).get_output(0)
det2_fp32.dtype = trt.DataType.FLOAT

det3 = network.get_layer(292).get_output(0)
det3_fp32 = network.add_identity(det3).get_output(0)
det3_fp32.dtype = trt.DataType.FLOAT
