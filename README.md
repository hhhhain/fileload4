# Copyright Huawei Technologies Co., Ltd. 2010-2018. All rights reserved.
# coding=utf-8/# -*- coding: utf-8 -*-

import time
import subprocess
import os
from fastapi import FastAPI, File, Form
import uvicorn
from log4py import Logger
from typing import List
import psutil
import threading
import asyncio  # 在文件顶部添加

Logger.set_level("INFO")
log = Logger.get_logger(__name__)

app = FastAPI()


def set_cpu():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        os.environ["cpu_usage"] = str(cpu_usage)


def get_cpu():
    cpu_usage = float(os.environ.get("cpu_usage"))
    return cpu_usage


# 在后台线程中加载模型
thread = threading.Thread(target=set_cpu)
thread.start()


@app.on_event("startup")
def load_model():
    from customize_service import videoSearchAIService
    # cp_engine_file_path = os.environ.get("cp_engine_file_path", "weights/CP26classes_epoch_180_t4_batch10.trt")
    cp_engine_file_path = os.environ.get("cp_engine_file_path", "/home/ma-user/infer/video-deal-search/video-deal-service/pt/CP26classes_epoch_180_fp16_bs10_640_1088_fromexec.trt")
    screen_engine_file_path = os.environ.get("screen_engine_file_path", "/home/ma-user/infer/video-deal-search/video-deal-service/pt/CP26classes_epoch_180_fp16_bs10_640_1088_fromexec.trt")
    pose_engine_file_path = os.environ.get("pose_engine_file_path", "weights/yolov8s-pose-h640w1088-fp16.trt") # 14280
    video_search = videoSearchAIService(cp_engine_file_path, screen_engine_file_path, pose_engine_file_path)
    app.state.model = video_search


async def get_detect_result(request, cpu_usage):
    data = request
    pid = os.getpid()
    start_time = time.time()
    video_search = app.state.model
    data = video_search.preprocess(data)
    data = video_search.inference(data, cpu_usage)
    result = video_search.postprocess(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    log.info("pid: {}, all cost: {}s, trace_id: {}".format(pid, elapsed_time, data["is_screen"]))
    return result


@app.post('/detect')
async def detect(files: List[bytes] = File(...), p_person_phone_thres: str = Form(...),
                 p_phone_hand_thres: str = Form(...),
                 c_person_notebook_thres: str = Form(...), c_notebook_hands_thres: str = Form(...),
                 is_screen: str = Form(...),
                 sceneType: str = Form(...)):
    data = {}
    data["p_person_phone_thres"] = p_person_phone_thres
    data["p_phone_hand_thres"] = p_phone_hand_thres
    data["c_person_notebook_thres"] = c_person_notebook_thres
    data["c_notebook_hands_thres"] = c_notebook_hands_thres
    data["sceneType"] = sceneType
    data["is_screen"] = is_screen
    data["files"] = files
    log.info(f"sceneType : {sceneType}")
    cpu_usage = get_cpu()
    result = await get_detect_result(data, cpu_usage)
    return result



if __name__ == '__main__':
    # try:
    #     workers = int(os.environ.get("workers", "2"))
    #     uvicorn.run(app="inference:app", host="0.0.0.0", port=8080, workers=workers)
    # except Exception as error:
    #     log.error(error)
    # 手动加载模型
    load_model()

    # 测试数据准备（替换为实际路径）
    test_file_paths = [f'image/test{i}.jpeg' for i in range(1, 11)]  # 10张图片路径
    test_files = []
    for path in test_file_paths:
        with open(path, 'rb') as f:
            test_files.append(f.read())

    test_params = {
        "p_person_phone_thres": "0.6",
        "p_phone_hand_thres": "0.6",
        "c_person_notebook_thres": "0.6",
        "c_notebook_hands_thres": "0.6",
        "is_screen": "False",
        "sceneType": "office",
        "files": test_files  # 包含10张图片的列表
    }

    # 模拟CPU使用率（可选）
    cpu_usage = 0.0

    # 批量处理并输出结果
    #results = get_detect_result(test_params, cpu_usage)
    results = asyncio.run(get_detect_result(test_params, cpu_usage))

    log.info(f"图片的处理结果：{results}")
    # for i, result in enumerate(results):
    #     log.info(f"图片{i + 1}的处理结果：{result}")
    
    # try:
    #     workers = int(os.environ.get("workers", "2"))
    #     uvicorn.run(app="inference:app", host="0.0.0.0", port=8080, workers=workers)
    # except Exception as error:
    #     log.error(error)
