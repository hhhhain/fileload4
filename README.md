// batch predict
  for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
    // Get a batch of images
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
      cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
      img_batch.push_back(img);
      img_name_batch.push_back(file_names[j]);
    }

    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);

    // Draw bounding boxes
    draw_bbox(img_batch, res_batch);

    // Save images
    for (size_t j = 0; j < img_batch.size(); j++) {
      cv::imwrite("_" + img_name_batch[j], img_batch[j]);
    }
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  cuda_preprocess_destroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  // Print histogram of the output distribution
  // std::cout << "\nOutput:\n\n";
  // for (unsigned int i = 0; i < kOutputSize; i++) {
  //   std::cout << prob[i] << ", ";
  //   if (i % 10 == 0) std::cout << std::endl;
  // }
  // std::cout << std::endl;

  return 0;
}
