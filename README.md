        dummy = np.random.rand(10, 3, 640, 1088).astype(np.float16)
        # run a couple times to ensure kernels compiled & memory allocated
        for _ in range(20):
            _outs = self.infer(dummy, "detection")
        cuda.Context.synchronize()
        print("warm up done.")        
        exit()
