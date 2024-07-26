import time
import math
import cv2
import requests
import json
import numpy as np
import os
import pandas as pd
import datetime
import multiprocessing as mp
import psutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def encode_image(img_rgb):
    img_bytes = str(cv2.imencode('.jpg', img_rgb)[1].tobytes())
    return img_bytes


def decode_image(img_bytes):
    img_jpg = np.frombuffer(eval(img_bytes), dtype=np.uint8)
    img_rgb = np.array(cv2.imdecode(img_jpg, cv2.IMREAD_UNCHANGED))
    return img_rgb

def sfg_get_next_init_task(
    job_uid=None,
    video_cap=None,
    video_conf=None,
    curr_cam_frame_id=None,
    curr_conf_frame_id=None
):
    assert video_cap

    # 从视频流读取一帧，根据fps跳帧
    cam_fps = video_cap.get(cv2.CAP_PROP_FPS)
    conf_fps = min(video_conf['fps'], cam_fps)

    frame = None
    new_cam_frame_id = None
    new_conf_frame_id = None
    while True:
        # 从video_fps中实际读取
        cam_frame_id = video_cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = video_cap.read()
        if not ret:
            time.sleep(1)
            continue

        assert ret

        conf_frame_id = math.floor((conf_fps * 1.0 / cam_fps) * cam_frame_id)
        if conf_frame_id != curr_conf_frame_id:
            # 提高fps时，conf_frame_id 远大于 curr_conf_frame_id
            # 降低fps时，conf_frame_id 远小于 curr_conf_frame_id
            # 持平fps时，conf_frame_id 最多为 curr_conf_frame_id + 1
            new_cam_frame_id = cam_frame_id
            new_conf_frame_id = conf_frame_id
            break

    # print("cam_fps={} conf_fps={}".format(cam_fps, conf_fps))
    # print("new_cam_frame_id={} new_conf_frame_id={}".format(new_cam_frame_id, new_conf_frame_id))



    input_ctx = dict()
    # input_ctx['image'] = (video_cap.get(cv2.CAP_PROP_POS_FRAMES), numpy.array(frame).shape)
    st_time = time.time()
    input_ctx['image'] = encode_image(frame)
    ed_time = time.time()


    return new_cam_frame_id, new_conf_frame_id, input_ctx




if __name__ == '__main__':
    # 0、初始化数据流来源（TODO：从缓存区读取）
    cap = cv2.VideoCapture('traffic-720p.mp4')
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    
    n = 0
    curr_cam_frame_id = 0
    curr_conf_frame_id = 0

    cur_plan_list = [{'encoder': 'JPEG', 'fps': 1, 'reso': '360p'},
                     {'encoder': 'JPEG', 'fps': 2, 'reso': '360p'},
                     {'encoder': 'JPEG', 'fps': 5, 'reso': '360p'},
                     {'encoder': 'JPEG', 'fps': 10, 'reso': '360p'},
                     {'encoder': 'JPEG', 'fps': 15, 'reso': '360p'},
                     {'encoder': 'JPEG', 'fps': 25, 'reso': '360p'},
                     ]
    # 逐帧汇报结果，逐帧汇报运行时情境
    while True:
        # print("In main: curr_cam_frame_id:{}, curr_conf_frame_id:{}".format(curr_cam_frame_id, curr_conf_frame_id))
        # 1、根据video_conf，获取本次循环的输入数据（TODO：从缓存区读取）
        cam_frame_id, conf_frame_id, output_ctx = \
            sfg_get_next_init_task(job_uid=1,
                                    video_cap=cap,
                                    video_conf=cur_plan_list[(n // 2) %6],
                                    curr_cam_frame_id=curr_cam_frame_id,
                                    curr_conf_frame_id=curr_conf_frame_id)
        print((n // 2) %6)
        print("In main: cam_frame_id:{}, conf_frame_id:{}".format(cam_frame_id, conf_frame_id))

        curr_cam_frame_id = cam_frame_id
        curr_conf_frame_id = conf_frame_id
        
        n += 1
        
        time.sleep(3)
    

