  printf("222222222222222222222222\n");
  for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
    const auto& yolo = mYoloKernel[i];
    // grid数乘以bs
    numElem = yolo.width * yolo.height * batchSize;
    // 每个cell对应一个线程，如果线程大于总cell数了，那就降低线程的数量。而mThreadCount一般都比numElem小得多。
    if (numElem < mThreadCount) mThreadCount = numElem;
    printf("333333333333333333\n");
    // 这里的用法是cuda kernel的调用。假设用__global__定义了一个函数fun。那么执行的时候就必须这样调用执行 fun<<<(四个内核参数)>>>(自己定义的输入参数)
    // 这里的内核参数是控制线程数的，分别是<<<gridsize, blocksize, sharedmem, stream>>>。
    // 先看第二个参数，blocksize决定了每个block能处理多少个cell，这里设置为mThreadCount，已知一般情况下mThreadCount都比numElem小得多，需要计算到底需要多少个mThreadCount
    // 第一个参数是向上取整的写法。计算出需要多少个block。这样保证了每个cell都有一个线程进行处理。
    CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
      (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem, is_segmentation_);
  }
