2025-09-14 14:22:33 INFO object_detect.__init__(object_detect.py:180): bingding:images,(10, 3, 640, 1088)
[09/14/2025-14:22:33] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
host_mem is 20889600
add input done
2025-09-14 14:22:33 INFO object_detect.__init__(object_detect.py:180): bingding:output0,(10, 93, 80, 136)
[09/14/2025-14:22:33] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
host_mem is 10118400
add output done
2025-09-14 14:22:33 INFO object_detect.__init__(object_detect.py:180): bingding:358,(10, 93, 40, 68)
[09/14/2025-14:22:33] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
host_mem is 2529600
add output done
2025-09-14 14:22:33 INFO object_detect.__init__(object_detect.py:180): bingding:359,(10, 93, 20, 34)
[09/14/2025-14:22:33] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
host_mem is 632400
add output done
2025-09-14 14:22:33 INFO object_detect.__init__(object_detect.py:180): bingding:yolo_out_post,(10, 38001, 1, 1)


    inputs = det_tensors
    network.unmark_output(inputs)
