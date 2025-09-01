# fileload4





class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()







Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::validateInputBindings::2083, condition: profileMinDims.d[i] <= dimensions.d[i]. Supplied binding dimension [1,3,640,1088] for bindings[0] exceed min ~ max range at index 0, maximum dimension in profile is 10, minimum dimension in profile is 10, but supplied dimension is 1.




Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv: 5.63274ms
/model.0/conv/Conv: 2.36189ms
Reformatting CopyNode for Input Tensor 0 to PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul): 0ms
PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul): 0.939456ms
Reformatting CopyNode for Input Tensor 0 to /model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), /model.1/act/Mul): 0ms
/model.1/conv/Conv + PWN(PWN(/model.1/act/Sigmoid), /model.1/act/Mul): 1.32118ms
/model.2/cv1/conv/Conv || /model.2/cv2/conv/Conv: 0.984832ms
PWN(PWN(/model.2/cv1/act/Sigmoid), /model.2/cv1/act/Mul): 0.256256ms
/model.2/m/m.0/cv1/conv/Conv + PWN(PWN(/model.2/m/m.0/cv1/act/Sigmoid), /model.2/m/m.0/cv1/act/Mul): 0.439968ms
/model.2/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.2/m/m.0/cv2/act/Sigmoid), /model.2/m/m.0/cv2/act/Mul), /model.2/m/m.0/Add): 1.1937ms
PWN(PWN(/model.2/cv2/act/Sigmoid), /model.2/cv2/act/Mul): 0.43216ms
/model.2/m/m.0/Add_output_0 copy: 0.279168ms
/model.2/cv3/conv/Conv + PWN(PWN(/model.2/cv3/act/Sigmoid), /model.2/cv3/act/Mul): 1.00554ms
/model.3/conv/Conv + PWN(PWN(/model.3/act/Sigmoid), /model.3/act/Mul): 0.9624ms
/model.4/cv1/conv/Conv || /model.4/cv2/conv/Conv: 0.472896ms
PWN(PWN(/model.4/cv1/act/Sigmoid), /model.4/cv1/act/Mul): 0.131456ms
/model.4/m/m.0/cv1/conv/Conv + PWN(PWN(/model.4/m/m.0/cv1/act/Sigmoid), /model.4/m/m.0/cv1/act/Mul): 0.26416ms
/model.4/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.4/m/m.0/cv2/act/Sigmoid), /model.4/m/m.0/cv2/act/Mul), /model.4/m/m.0/Add): 0.768ms
/model.4/m/m.1/cv1/conv/Conv + PWN(PWN(/model.4/m/m.1/cv1/act/Sigmoid), /model.4/m/m.1/cv1/act/Mul): 0.26624ms
/model.4/m/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.4/m/m.1/cv2/act/Sigmoid), /model.4/m/m.1/cv2/act/Mul), /model.4/m/m.1/Add): 0.75776ms
PWN(PWN(/model.4/cv2/act/Sigmoid), /model.4/cv2/act/Mul): 0.214496ms
/model.4/m/m.1/Add_output_0 copy: 0.143904ms
/model.4/cv3/conv/Conv + PWN(PWN(/model.4/cv3/act/Sigmoid), /model.4/cv3/act/Mul): 0.378304ms
/model.5/conv/Conv + PWN(PWN(/model.5/act/Sigmoid), /model.5/act/Mul): 0.881216ms
/model.6/cv1/conv/Conv || /model.6/cv2/conv/Conv: 18.1158ms
PWN(PWN(/model.6/cv1/act/Sigmoid), /model.6/cv1/act/Mul): 0.086848ms
/model.6/m/m.0/cv1/conv/Conv + PWN(PWN(/model.6/m/m.0/cv1/act/Sigmoid), /model.6/m/m.0/cv1/act/Mul): 0.124928ms
/model.6/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.6/m/m.0/cv2/act/Sigmoid), /model.6/m/m.0/cv2/act/Mul), /model.6/m/m.0/Add): 0.54192ms
/model.6/m/m.1/cv1/conv/Conv + PWN(PWN(/model.6/m/m.1/cv1/act/Sigmoid), /model.6/m/m.1/cv1/act/Mul): 0.125472ms
/model.6/m/m.1/cv2/conv/Conv + PWN(PWN(PWN(/model.6/m/m.1/cv2/act/Sigmoid), /model.6/m/m.1/cv2/act/Mul), /model.6/m/m.1/Add): 0.54208ms
/model.6/m/m.2/cv1/conv/Conv + PWN(PWN(/model.6/m/m.2/cv1/act/Sigmoid), /model.6/m/m.2/cv1/act/Mul): 0.123776ms
/model.6/m/m.2/cv2/conv/Conv + PWN(PWN(PWN(/model.6/m/m.2/cv2/act/Sigmoid), /model.6/m/m.2/cv2/act/Mul), /model.6/m/m.2/Add): 0.54272ms
PWN(PWN(/model.6/cv2/act/Sigmoid), /model.6/cv2/act/Mul): 0.069408ms
/model.6/m/m.2/Add_output_0 copy: 0.078048ms
/model.6/cv3/conv/Conv + PWN(PWN(/model.6/cv3/act/Sigmoid), /model.6/cv3/act/Mul): 0.270336ms
/model.7/conv/Conv + PWN(PWN(/model.7/act/Sigmoid), /model.7/act/Mul): 0.906272ms
/model.8/cv1/conv/Conv || /model.8/cv2/conv/Conv: 0.2488ms
PWN(PWN(/model.8/cv1/act/Sigmoid), /model.8/cv1/act/Mul): 0.0528ms
/model.8/m/m.0/cv1/conv/Conv + PWN(PWN(/model.8/m/m.0/cv1/act/Sigmoid), /model.8/m/m.0/cv1/act/Mul): 0.088416ms
/model.8/m/m.0/cv2/conv/Conv + PWN(PWN(PWN(/model.8/m/m.0/cv2/act/Sigmoid), /model.8/m/m.0/cv2/act/Mul), /model.8/m/m.0/Add): 0.507776ms
PWN(PWN(/model.8/cv2/act/Sigmoid), /model.8/cv2/act/Mul): 0.051424ms
/model.8/m/m.0/Add_output_0 copy: 0.045056ms
/model.8/cv3/conv/Conv + PWN(PWN(/model.8/cv3/act/Sigmoid), /model.8/cv3/act/Mul): 0.24576ms
/model.9/cv1/conv/Conv + PWN(PWN(/model.9/cv1/act/Sigmoid), /model.9/cv1/act/Mul): 0.133152ms
/model.9/m/MaxPool: 0.137184ms
/model.9/m_1/MaxPool: 0.137216ms
/model.9/m_2/MaxPool: 0.137216ms
/model.9/cv1/act/Mul_output_0 copy: 0.045088ms
/model.9/m/MaxPool_output_0 copy: 0.044352ms
/model.9/m_1/MaxPool_output_0 copy: 0.04368ms
/model.9/cv2/conv/Conv + PWN(PWN(/model.9/cv2/act/Sigmoid), /model.9/cv2/act/Mul): 0.438272ms
/model.10/conv/Conv + PWN(PWN(/model.10/act/Sigmoid), /model.10/act/Mul): 0.132736ms
/model.11/Resize: 0.082176ms
/model.11/Resize_output_0 copy: 0.14656ms
/model.13/cv1/conv/Conv || /model.13/cv2/conv/Conv: 0.443392ms
PWN(PWN(/model.13/cv1/act/Sigmoid), /model.13/cv1/act/Mul): 0.087264ms
/model.13/m/m.0/cv1/conv/Conv + PWN(PWN(/model.13/m/m.0/cv1/act/Sigmoid), /model.13/m/m.0/cv1/act/Mul): 0.125248ms
/model.13/m/m.0/cv2/conv/Conv + PWN(PWN(/model.13/m/m.0/cv2/act/Sigmoid), /model.13/m/m.0/cv2/act/Mul): 0.483552ms
PWN(PWN(/model.13/cv2/act/Sigmoid), /model.13/cv2/act/Mul): 0.06992ms
/model.13/cv3/conv/Conv + PWN(PWN(/model.13/cv3/act/Sigmoid), /model.13/cv3/act/Mul): 0.270144ms
/model.14/conv/Conv + PWN(PWN(/model.14/act/Sigmoid), /model.14/act/Mul): 0.163904ms
/model.15/Resize: 0.159168ms
/model.15/Resize_output_0 copy: 0.278272ms
/model.17/cv1/conv/Conv || /model.17/cv2/conv/Conv: 0.518048ms
PWN(PWN(/model.17/cv1/act/Sigmoid), /model.17/cv1/act/Mul): 0.132128ms
/model.17/m/m.0/cv1/conv/Conv + PWN(PWN(/model.17/m/m.0/cv1/act/Sigmoid), /model.17/m/m.0/cv1/act/Mul): 0.265632ms
/model.17/m/m.0/cv2/conv/Conv + PWN(PWN(/model.17/m/m.0/cv2/act/Sigmoid), /model.17/m/m.0/cv2/act/Mul): 0.616192ms
PWN(PWN(/model.17/cv2/act/Sigmoid), /model.17/cv2/act/Mul): 0.213824ms
/model.17/cv3/conv/Conv + PWN(PWN(/model.17/cv3/act/Sigmoid), /model.17/cv3/act/Mul): 0.372736ms
/model.18/conv/Conv + PWN(PWN(/model.18/act/Sigmoid), /model.18/act/Mul): 0.501376ms
/model.14/act/Mul_output_0 copy: 0.077536ms
/model.20/cv1/conv/Conv || /model.20/cv2/conv/Conv: 0.264864ms
PWN(PWN(/model.20/cv1/act/Sigmoid), /model.20/cv1/act/Mul): 0.08768ms
/model.20/m/m.0/cv1/conv/Conv + PWN(PWN(/model.20/m/m.0/cv1/act/Sigmoid), /model.20/m/m.0/cv1/act/Mul): 0.123264ms
/model.20/m/m.0/cv2/conv/Conv + PWN(PWN(/model.20/m/m.0/cv2/act/Sigmoid), /model.20/m/m.0/cv2/act/Mul): 0.483328ms
PWN(PWN(/model.20/cv2/act/Sigmoid), /model.20/cv2/act/Mul): 0.069664ms
/model.20/cv3/conv/Conv + PWN(PWN(/model.20/cv3/act/Sigmoid), /model.20/cv3/act/Mul): 0.270304ms
/model.21/conv/Conv + PWN(PWN(/model.21/act/Sigmoid), /model.21/act/Mul): 0.476448ms
/model.10/act/Mul_output_0 copy: 0.043744ms
/model.23/cv1/conv/Conv || /model.23/cv2/conv/Conv: 0.24304ms
PWN(PWN(/model.23/cv1/act/Sigmoid), /model.23/cv1/act/Mul): 0.051584ms
/model.23/m/m.0/cv1/conv/Conv + PWN(PWN(/model.23/m/m.0/cv1/act/Sigmoid), /model.23/m/m.0/cv1/act/Mul): 0.089376ms
/model.23/m/m.0/cv2/conv/Conv + PWN(PWN(/model.23/m/m.0/cv2/act/Sigmoid), /model.23/m/m.0/cv2/act/Mul): 0.46592ms
PWN(PWN(/model.23/cv2/act/Sigmoid), /model.23/cv2/act/Mul): 0.0512ms
/model.23/cv3/conv/Conv + PWN(PWN(/model.23/cv3/act/Sigmoid), /model.23/cv3/act/Mul): 0.247168ms
/model.24/m.2/Conv: 0.107168ms
/model.24/Reshape_4 + /model.24/Transpose_2: 0.106464ms
PWN(/model.24/Sigmoid_2): 0.01424ms
/model.24/Split_2: 4.30416ms
/model.24/Split_2_14: 0.0152ms
/model.24/Split_2_15: 0.024576ms
/model.24/m.1/Conv: 0.220896ms
/model.24/Reshape_2 + /model.24/Transpose_1: 0.149792ms
PWN(/model.24/Sigmoid_1): 0.047104ms
/model.24/Split_1: 0.02864ms
/model.24/Split_1_10: 0.027872ms
/model.24/Split_1_11: 0.086816ms
/model.24/m.0/Conv: 0.456704ms
/model.24/Reshape + /model.24/Transpose: 0.583232ms
PWN(/model.24/Sigmoid): 0.254432ms
/model.24/Split: 0.09216ms
/model.24/Split_6: 0.087072ms
/model.24/Split_7: 0.33072ms
/model.24/Constant_22_output_0: 0ms
PWN(PWN(/model.24/Constant_20_output_0 + (Unnamed Layer* 284) [Shuffle] + /model.24/Mul_10, PWN(/model.24/Constant_21_output_0 + (Unnamed Layer* 287) [Shuffle], /model.24/Pow_2)), /model.24/Mul_11): 0.008192ms
/model.24/Constant_18_output_0: 0ms
PWN(PWN(/model.24/Constant_17_output_0 + (Unnamed Layer* 276) [Shuffle] + /model.24/Mul_8, /model.24/Add_2), /model.24/Constant_19_output_0 + (Unnamed Layer* 281) [Shuffle] + /model.24/Mul_9): 0.00704ms
/model.24/Mul_9_output_0 copy: 2.57437ms
/model.24/Mul_11_output_0 copy: 0.020448ms
/model.24/Reshape_5: 0ms
/model.24/Reshape_5_copy_output: 4.69021ms
/model.24/Constant_14_output_0: 0ms
PWN(PWN(/model.24/Constant_12_output_0 + (Unnamed Layer* 259) [Shuffle] + /model.24/Mul_6, PWN(/model.24/Constant_13_output_0 + (Unnamed Layer* 262) [Shuffle], /model.24/Pow_1)), /model.24/Mul_7): 0.0104ms
/model.24/Constant_10_output_0: 0ms
PWN(PWN(/model.24/Constant_9_output_0 + (Unnamed Layer* 251) [Shuffle] + /model.24/Mul_4, /model.24/Add_1), /model.24/Constant_11_output_0 + (Unnamed Layer* 256) [Shuffle] + /model.24/Mul_5): 0.010144ms
/model.24/Mul_5_output_0 copy: 0.120928ms
/model.24/Mul_7_output_0 copy: 0.034816ms
/model.24/Reshape_3: 0ms
/model.24/Reshape_3_copy_output: 0.044288ms
/model.24/Constant_6_output_0: 0ms
PWN(PWN(/model.24/Constant_4_output_0 + (Unnamed Layer* 234) [Shuffle] + /model.24/Mul_2, PWN(/model.24/Constant_5_output_0 + (Unnamed Layer* 237) [Shuffle], /model.24/Pow)), /model.24/Mul_3): 0.020608ms
/model.24/Constant_2_output_0: 0ms
PWN(PWN(/model.24/Constant_1_output_0 + (Unnamed Layer* 226) [Shuffle] + /model.24/Mul, /model.24/Add), /model.24/Constant_3_output_0 + (Unnamed Layer* 231) [Shuffle] + /model.24/Mul_1): 0.02624ms
/model.24/Mul_1_output_0 copy: 0.122976ms
/model.24/Mul_3_output_0 copy: 0.124896ms
/model.24/Reshape_1: 0ms
/model.24/Reshape_1_copy_output: 0.167072ms
infer: 68.08 ms














Reformatting CopyNode for Input Tensor 0 to Conv_0: 1.12029ms
Conv_0: 3.52794ms
PWN(PWN(Sigmoid_1), Mul_2): 0.942848ms
Conv_3 + PWN(PWN(Sigmoid_4), Mul_5): 1.33309ms
Conv_6 || Conv_16: 0.973984ms
PWN(PWN(Sigmoid_7), Mul_8): 0.26432ms
PWN(PWN(Sigmoid_17), Mul_18): 0.434816ms
Conv_9 + PWN(PWN(Sigmoid_10), Mul_11): 0.44ms
Conv_12: 0.852352ms
PWN(PWN(PWN(Sigmoid_13), Mul_14), Add_15): 0.39472ms
Conv_20 + PWN(PWN(Sigmoid_21), Mul_22): 0.983776ms
Conv_23 + PWN(PWN(Sigmoid_24), Mul_25): 0.960512ms
Conv_26 || Conv_43: 0.47312ms
PWN(PWN(Sigmoid_27), Mul_28): 0.133088ms
PWN(PWN(Sigmoid_44), Mul_45): 0.219136ms
Conv_29 + PWN(PWN(Sigmoid_30), Mul_31): 0.26624ms
Conv_32: 0.624672ms
PWN(PWN(PWN(Sigmoid_33), Mul_34), Add_35): 0.186304ms
Conv_36 + PWN(PWN(Sigmoid_37), Mul_38): 0.266272ms
Conv_39: 0.622112ms
PWN(PWN(PWN(Sigmoid_40), Mul_41), Add_42): 0.321088ms
Conv_47 + PWN(PWN(Sigmoid_48), Mul_49): 0.387008ms
Conv_50 + PWN(PWN(Sigmoid_51), Mul_52): 0.901728ms
Conv_53 || Conv_77: 0.262528ms
PWN(PWN(Sigmoid_54), Mul_55): 0.07168ms
PWN(PWN(Sigmoid_78), Mul_79): 0.087744ms
Conv_56 + PWN(PWN(Sigmoid_57), Mul_58): 0.12464ms
Conv_59: 0.479872ms
PWN(PWN(PWN(Sigmoid_60), Mul_61), Add_62): 0.126944ms
Conv_63 + PWN(PWN(Sigmoid_64), Mul_65): 0.1264ms
Conv_66: 0.47984ms
PWN(PWN(PWN(Sigmoid_67), Mul_68), Add_69): 0.126944ms
Conv_70 + PWN(PWN(Sigmoid_71), Mul_72): 0.12864ms
Conv_73: 0.479648ms
PWN(PWN(PWN(Sigmoid_74), Mul_75), Add_76): 0.13312ms
Conv_81 + PWN(PWN(Sigmoid_82), Mul_83): 0.271648ms
Conv_84 + PWN(PWN(Sigmoid_85), Mul_86): 0.890656ms
Conv_87 || Conv_97: 0.242592ms
PWN(PWN(Sigmoid_88), Mul_89): 0.04096ms
PWN(PWN(Sigmoid_98), Mul_99): 0.05328ms
Conv_90 + PWN(PWN(Sigmoid_91), Mul_92): 0.091968ms
Conv_93: 0.464224ms
PWN(PWN(PWN(Sigmoid_94), Mul_95), Add_96): 0.055392ms
Conv_101 + PWN(PWN(Sigmoid_102), Mul_103): 0.246528ms
Conv_104 + PWN(PWN(Sigmoid_105), Mul_106): 0.13312ms
MaxPool_107: 0.14032ms
MaxPool_108: 0.138208ms
MaxPool_109: 0.138912ms
228 copy: 0.061216ms
229 copy: 0.059936ms
230 copy: 0.061056ms
Conv_111 + PWN(PWN(Sigmoid_112), Mul_113): 0.43664ms
Conv_114 + PWN(PWN(Sigmoid_115), Mul_116): 0.134208ms
Resize_118: 0.086496ms
243 copy: 0.2032ms
Conv_120 || Conv_129: 0.439584ms
PWN(PWN(Sigmoid_121), Mul_122): 0.071424ms
PWN(PWN(Sigmoid_130), Mul_131): 0.086912ms
Conv_123 + PWN(PWN(Sigmoid_124), Mul_125): 0.125056ms
Conv_126 + PWN(PWN(Sigmoid_127), Mul_128): 0.49968ms
Conv_133 + PWN(PWN(Sigmoid_134), Mul_135): 0.270368ms
Conv_136 + PWN(PWN(Sigmoid_137), Mul_138): 0.165792ms
Resize_140: 0.15984ms
268 copy: 0.395232ms
Conv_142 || Conv_151: 0.51376ms
PWN(PWN(Sigmoid_143), Mul_144): 0.133408ms
PWN(PWN(Sigmoid_152), Mul_153): 0.219136ms
Conv_145 + PWN(PWN(Sigmoid_146), Mul_147): 0.26624ms
Conv_148 + PWN(PWN(Sigmoid_149), Mul_150): 0.61808ms
Conv_155 + PWN(PWN(Sigmoid_156), Mul_157): 0.389568ms
Conv_158 + PWN(PWN(Sigmoid_159), Mul_160): 0.501408ms
Conv_198: 0.456544ms
263 copy: 0.107008ms
Conv_162 || Conv_171: 0.26416ms
Reformatting CopyNode for Input Tensor 0 to Reshape_212 + Transpose_213: 0.256ms
Reshape_212 + Transpose_213: 0.86224ms
PWN(PWN(Sigmoid_163), Mul_164): 0.070656ms
PWN(PWN(Sigmoid_172), Mul_173): 0.087008ms
Conv_165 + PWN(PWN(Sigmoid_166), Mul_167): 0.128032ms
PWN(Sigmoid_214): 0.172768ms
Split_215: 0.092416ms
Split_215_0: 0.088096ms
Split_215_1: 0.333792ms
Conv_168 + PWN(PWN(Sigmoid_169), Mul_170): 0.497696ms
PWN(PWN(350 + (Unnamed Layer* 219) [Shuffle] + Mul_217, Add_219), 354 + (Unnamed Layer* 224) [Shuffle] + Mul_221): 0.026592ms
PWN(PWN(356 + (Unnamed Layer* 227) [Shuffle] + Mul_223, PWN(358 + (Unnamed Layer* 230) [Shuffle], Pow_225)), Mul_227): 0.02848ms
355 copy: 0.140832ms
361 copy: 0.135872ms
349 copy: 0.399328ms
Reshape_235: 0.003424ms
Conv_175 + PWN(PWN(Sigmoid_176), Mul_177): 0.2728ms
Conv_178 + PWN(PWN(Sigmoid_179), Mul_180): 0.475392ms
Conv_236: 0.208544ms
238 copy: 0.061504ms
Conv_182 || Conv_191: 0.244ms
Reformatting CopyNode for Input Tensor 0 to Reshape_250 + Transpose_251: 0.07168ms
Reshape_250 + Transpose_251: 0.167488ms
PWN(PWN(Sigmoid_183), Mul_184): 0.03936ms
PWN(PWN(Sigmoid_192), Mul_193): 0.05296ms
Conv_185 + PWN(PWN(Sigmoid_186), Mul_187): 0.09168ms
PWN(Sigmoid_252): 0.04992ms
Split_253: 0.030752ms
Split_253_2: 0.02848ms
Split_253_3: 0.088224ms
Conv_188 + PWN(PWN(Sigmoid_189), Mul_190): 0.466944ms
PWN(PWN(396 + (Unnamed Layer* 266) [Shuffle] + Mul_255, Add_257), 400 + (Unnamed Layer* 271) [Shuffle] + Mul_259): 0.012096ms
PWN(PWN(402 + (Unnamed Layer* 274) [Shuffle] + Mul_261, PWN(404 + (Unnamed Layer* 277) [Shuffle], Pow_263)), Mul_265): 0.010432ms
401 copy: 0.073728ms
407 copy: 0.036864ms
395 copy: 0.106496ms
Reshape_273: 0.003456ms
Conv_195 + PWN(PWN(Sigmoid_196), Mul_197): 0.24768ms
Conv_274: 0.108384ms
Reformatting CopyNode for Input Tensor 0 to Reshape_288 + Transpose_289: 0.023232ms
Reshape_288 + Transpose_289: 0.040832ms
PWN(Sigmoid_290): 0.014688ms
Split_291: 0.010208ms
Split_291_4: 0.009472ms
Split_291_5: 0.025376ms
PWN(PWN(442 + (Unnamed Layer* 313) [Shuffle] + Mul_293, Add_295), 446 + (Unnamed Layer* 318) [Shuffle] + Mul_297): 0.008224ms
PWN(PWN(448 + (Unnamed Layer* 321) [Shuffle] + Mul_299, PWN(450 + (Unnamed Layer* 324) [Shuffle], Pow_301)), Mul_303): 0.009536ms
447 copy: 0.010944ms
453 copy: 0.012256ms
441 copy: 0.026624ms
Reshape_311: 0.003808ms
371 copy: 0.44592ms
417 copy: 0.118752ms
463 copy: 0.03136ms
infer: 36.90 ms

















import tensorrt as trt
from pathlib import Path

# 1. 设置 TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# 2. 指定 engine 文件路径
engine_file = "/home/ma-user/work/video-deal-service/weights/CP26classes_epoch_180_fp16_bs10_640_1088_fromAPI.trt"
assert Path(engine_file).exists(), f"Engine file not found: {engine_file}"

# 3. 反序列化 engine
with open(engine_file, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None, "Failed to deserialize engine"

print("Engine deserialized successfully!")
print(f"Number of bindings: {engine.num_bindings}")

# 4. 创建执行上下文
context = engine.create_execution_context()
assert context is not None, "Failed to create execution context"
print("Execution context created successfully!")

# 5. 打印每个 binding 的信息
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    dtype = engine.get_binding_dtype(i)
    shape = context.get_binding_shape(i)
    io_type = "Input" if engine.binding_is_input(i) else "Output"
    print(f"{io_type} -> Name: {name}, Index: {i}, Shape: {shape}, Dtype: {dtype}")













import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # 相当于 --verbose

engine_file = "/home/ma-user/work/video-deal-service/weights/CP26classes_epoch_180_fp16_bs10_640_1088_fromAPI.trt"

# 1. 读取engine文件
with open(engine_file, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. 创建执行上下文
context = engine.create_execution_context()

# 3. 分配输入输出显存
bindings = []
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)          # 返回 string
    dtype = trt.nptype(engine.get_binding_dtype(i))
    shape = context.get_binding_shape(i)       # 传 index
    io_type = "Input" if engine.binding_is_input(i) else "Output"
    print(f"{io_type} -> Name: {name}, Index: {i}, Shape: {shape}, Dtype: {dtype}")

# 4. 你可以手动拷贝输入数据到 device，然后运行推理
# context.execute_v2(bindings=bindings)   # 对应 enqueueV2






Input -> Name: images, Index: 0, Shape: (10, 3, 640, 1088), Dtype: <class 'numpy.float16'>
Output -> Name: output0, Index: 1, Shape: (10, 42840, 31), Dtype: <class 'numpy.float16'>
[08/31/2025-21:20:58] [TRT] [E] 1: [defaultAllocator.cpp::deallocate::35] Error Code 1: Cuda Runtime (invalid argument)
Segmentation fault


for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)          # 返回 string
    dtype = trt.nptype(engine.get_binding_dtype(i))
    shape = context.get_binding_shape(i)       # 传 index
    io_type = "Input" if engine.binding_is_input(i) else "Output"
    print(f"{io_type} -> Name: {name}, Index: {i}, Shape: {shape}, Dtype: {dtype}")







Traceback (most recent call last):
  File "test.py", line 22, in <module>
    shape = context.get_binding_shape(binding)
TypeError: get_binding_shape(): incompatible function arguments. The following argument types are supported:
    1. (self: tensorrt.tensorrt.IExecutionContext, binding: int) -> tensorrt.tensorrt.Dims

Invoked with: <tensorrt.tensorrt.IExecutionContext object at 0x7fce27c259b0>, 'images'
[08/31/2025-21:15:07] [TRT] [E] 1: [defaultAllocator.cpp::deallocate::35] Error Code 1: Cuda Runtime (invalid argument)
Segmentation fault






import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # 相当于 --verbose

engine_file = "your.engine"

# 1. 读取engine文件
with open(engine_file, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. 创建执行上下文
context = engine.create_execution_context()

# 3. 分配输入输出显存
bindings = []
for binding in engine:
    binding_idx = engine.get_binding_index(binding)
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    shape = context.get_binding_shape(binding)

    size = trt.volume(shape) * engine.max_batch_size  # 显存大小
    device_mem = cuda.mem_alloc(size * dtype().itemsize)
    bindings.append(int(device_mem))

    print(f"Binding: {binding}, Shape: {shape}, Dtype: {dtype}, Index: {binding_idx}")

# 4. 你可以手动拷贝输入数据到 device，然后运行推理
# context.execute_v2(bindings=bindings)   # 对应 enqueueV2







[08/31/2025-20:54:33] [TRT] [W] The enqueue() method has been deprecated when used with engines built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. Please use enqueueV2() instead.
[08/31/2025-20:54:33] [TRT] [W] Also, the batchSize argument passed into this function has no effect on changing the input shapes. Please use setBindingDimensions() function to change input shapes instead.





  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
      (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (2): C3(
      (cv1): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (4): C3(
      (cv1): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (5): Conv(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (6): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (7): Conv(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (8): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (9): SPPF(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    )
    (10): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (11): Upsample(scale_factor=2.0, mode='nearest')
    (12): Concat()
    (13): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (14): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (15): Upsample(scale_factor=2.0, mode='nearest')
    (16): Concat()
    (17): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (18): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (19): Concat()
    (20): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (21): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (22): Concat()
    (23): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (24): Detect(
      (m): ModuleList(
        (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
), 'ema': None, 'updates': None, 'optimizer': None, 'wandb_id': None, 'date': '2022-02-08T17:05:36.119232'}























  (model): Sequential(
    (0): Conv(
      (conv): Conv2d(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
      (bn): SyncBatchNorm(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (2): C3(
      (cv1): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (3): Conv(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (4): C3(
      (cv1): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (5): Conv(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (6): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (2): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (7): Conv(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (8): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (9): SPPF(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    )
    (10): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (11): Upsample(scale_factor=2.0, mode='nearest')
    (12): Concat()
    (13): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (14): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (15): Upsample(scale_factor=2.0, mode='nearest')
    (16): Concat()
    (17): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (18): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (19): Concat()
    (20): C3(
      (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (21): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (22): Concat()
    (23): C3(
      (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): SyncBatchNorm(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): SyncBatchNorm(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
    (24): Detect(
      (m): ModuleList(
        (0): Conv2d(128, 93, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(256, 93, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(512, 93, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)
