    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            # print(im.shape)
            # exit()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
            # print('nb is', nb)

        # Inference 替换tensorrt
        with dt[1]:
            # print()
            # print('im shape', im.shape)
            # exit()

            
            start_inf = time.time()            
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)
