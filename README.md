    std::cout << "Plugin constructor: mKernelCount = " << mKernelCount << std::endl;
    for (size_t i = 0; i < mYoloKernel.size(); ++i) {
        const auto& yolo = mYoloKernel[i];
        std::cout << "Head " << i 
                  << ": width=" << yolo.width 
                  << ", height=" << yolo.height 
                  << ", anchors=[";
        for (int j = 0; j < 6; ++j) {
            std::cout << yolo.anchors[j];
            if (j < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
