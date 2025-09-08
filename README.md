void YoloLayerPlugin::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize) {
  int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float); // 这条单张输出的大小，有个mMaxOutObject的限制，限制了最多输出多少个框。
  for (int idx = 0; idx < batchSize; ++idx) {
    // 注意看这里的idx在for循环里已经是bs的索引了，目的是把整个bs里的每张图的框计数res_count清零，表示还没写入任何框。
    printf("cudaMemsetAsync here\n");
    CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
  }
  int numElem = 0;
  printf("11 here\n");
  for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
    printf("22 here\n");
    const auto& yolo = mYoloKernel[i];
    printf("33 here\n");
    // grid数乘以bs
    numElem = yolo.width * yolo.height * batchSize;
    printf("44 here\n");
    // 每个cell对应一个线程，如果线程大于总cell数了，那就降低线程的数量。而mThreadCount一般都比numElem小得多。
    if (numElem < mThreadCount) mThreadCount = numElem;
    printf("55 here\n");
    // 这里的用法是cuda kernel的调用。假设用__global__定义了一个函数fun。那么执行的时候就必须这样调用执行 fun<<<(四个内核参数)>>>(自己定义的输入参数)
    // 这里的内核参数是控制线程数的，分别是<<<gridsize, blocksize, sharedmem, stream>>>。
    // 先看第二个参数，blocksize决定了每个block能处理多少个cell，这里设置为mThreadCount，已知一般情况下mThreadCount都比numElem小得多，需要计算到底需要多少个mThreadCount
    // 第一个参数是向上取整的写法。计算出需要多少个block。这样保证了每个cell都有一个线程进行处理。
    printf("mThreadCount is %d\n", mThreadCount);  
    printf("numElem is %d\n", numElem);
    CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
      (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem, is_segmentation_);
  }
}
我发现mThreadCount是0，能不能告诉我在哪给mThreadCount赋值，没有在参数栏传进来呀？
