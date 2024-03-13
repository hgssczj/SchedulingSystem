import time
import sys
import cv2
import requests
import json
import numpy as np
import os
import pandas as pd
import datetime
import multiprocessing as mp
import psutil
import importlib
import csv


resolution_wh = {
    "360p": {
        "w": 480,
        "h": 360
    },
    "480p": {
        "w": 640,
        "h": 480
    },
    "720p": {
        "w": 1280,
        "h": 720
    },
    "1080p": {
        "w": 1920,
        "h": 1080
    }
}


def work_func(task_name, q_input, q_output, model_ctx=None):
    task_name_split = task_name.split('_')  # 以下划线分割
    task_class_name = ''  # 任务对应的类名
    for name_split in task_name_split:
        temp_name = name_split.capitalize()   # 各个字符串的首字母大写
        task_class_name = task_class_name + temp_name
    print("启动服务",task_name)
    # 使用importlib动态加载任务对应的类
    path1 = os.path.abspath('.')
    path2 = os.path.join(path1, task_name)
    path3 = os.path.join(path2, '')
    sys.path.append(path3)   # 当前任务代码所在的文件夹，并将其加入模块导入搜索路径
    module_path = task_name + '.' + task_name  # 绝对导入
    module_name = importlib.import_module(module_path)  # 加载模块，例如face_detection
    class_obj = getattr(module_name, task_class_name)  # 从模块中加载执行类，例如FaceDetection

    # 创建执行类对应的对象
    if model_ctx is not None:
        work_obj = class_obj(model_ctx)
    else:
        work_obj = class_obj()

    # 开始监听，并循环执行任务
    
    while True:
        # 从队列中获取任务
        input_ctx = q_input.get()
        

        pid = os.getpid()  # 获得当前进程的pid
        p = psutil.Process(pid)  # 获取当前进程的Process对象


        # cpu占用的计算
        start_cpu_per = p.cpu_percent(interval=None)  # 任务执行前进程的cpu占用率
        # start_cpu_time = p.cpu_times()  # 任务执行前进程的cpu_time

        # 延时的计算
        start_time = time.time()  # 计时，统计延时

        # 执行任务
        output_ctx = work_obj(input_ctx)

        # 延时的计算
        end_time = time.time()

        # cpu占用的计算
        end_cpu_per = p.cpu_percent(interval=None)  # 任务执行后进程的cpu占用率
        # end_cpu_time = p.cpu_times()  # 任务执行后进程的cpu_time


        # 内存消耗的计算
        # end_memory = p.memory_full_info().uss  # 任务执行之后进程占用的内存值，每次执行任务之后都重新计算
        # after_data_memory = p.memory_info().data

    
        proc_resource_info = dict()  # 执行任务过程中的资源消耗情况
        proc_resource_info['pid'] = pid
        # 任务执行的cpu占用率，各个核上占用率之和的百分比->平均每个核的占用率，范围[0,1]
        proc_resource_info['cpu_util_use'] = end_cpu_per / 100 / psutil.cpu_count()
        proc_resource_info['mem_util_use'] = p.memory_percent(memtype='uss') / 100
        proc_resource_info['mem_use_amount'] = p.memory_full_info().uss
        proc_resource_info['compute_latency'] = end_time - start_time  # 任务执行的延时，单位：秒(s)
        if 'gpu_proc_time' in output_ctx:
            proc_resource_info['gpu_proc_time'] = output_ctx['gpu_proc_time']
        else:
            proc_resource_info['gpu_proc_time'] = 0
        
        if 'pre_proc_time' in output_ctx:
            proc_resource_info['pre_proc_time'] = output_ctx['pre_proc_time']
        else:
            proc_resource_info['pre_proc_time'] = 0
            
        if 'post_proc_time' in output_ctx:
            proc_resource_info['post_proc_time'] = output_ctx['post_proc_time']
        else:
            proc_resource_info['post_proc_time'] = 0

        output_ctx['proc_resource_info'] = proc_resource_info
        q_output.put(output_ctx)


def work_func_test(q_input, q_output):
    while True:
        # 从队列中获取任务
        input_ctx = q_input.get()
        

        pid = os.getpid()  # 获得当前进程的pid
        p = psutil.Process(pid)  # 获取当前进程的Process对象


        # cpu占用的计算
        start_cpu_per = p.cpu_percent(interval=None)  # 任务执行前进程的cpu占用率
        # start_cpu_time = p.cpu_times()  # 任务执行前进程的cpu_time

        # 延时的计算
        start_time = time.time()  # 计时，统计延时

        # 执行任务
        res = 0
        for i in range(1000000):
            res += 1
        
        time.sleep(0.005)
        
        res = 0
        for i in range(1000000):
            res += 1

        # 延时的计算
        end_time = time.time()

        # cpu占用的计算
        end_cpu_per = p.cpu_percent(interval=None)  # 任务执行后进程的cpu占用率
        # end_cpu_time = p.cpu_times()  # 任务执行后进程的cpu_time


        # 内存消耗的计算
        # end_memory = p.memory_full_info().uss  # 任务执行之后进程占用的内存值，每次执行任务之后都重新计算
        # after_data_memory = p.memory_info().data

    
        proc_resource_info = dict()  # 执行任务过程中的资源消耗情况
        proc_resource_info['pid'] = pid
        # 任务执行的cpu占用率，各个核上占用率之和的百分比->平均每个核的占用率，范围[0,1]
        proc_resource_info['cpu_util_use'] = end_cpu_per / 100 / psutil.cpu_count()
        proc_resource_info['mem_util_use'] = p.memory_percent(memtype='uss') / 100
        proc_resource_info['compute_latency'] = end_time - start_time  # 任务执行的延时，单位：秒(s)

        output_ctx = {}
        output_ctx['proc_resource_info'] = proc_resource_info
        q_output.put(output_ctx)

'''
if __name__ == '__main__':
    # 测试单独在节点上运行服务时服务的执行情况（资源占用率、延时等），与服务运行在整个调度系统上时的执行情况是否一致
    # 此版本的主函数用于测试face_detection服务
    json_data='\
    {\
        "name": "face_pose_estimation",  \
        "flow": ["face_detection", "face_alignment"],\
        "model_ctx": {  \
            "face_detection": {\
                "net_type": "mb_tiny_RFB_fd",\
                "input_size": 640,\
                "threshold": 0.7,\
                "candidate_size": 1500,\
                "device": "cpu"\
            },\
            "face_alignment": {\
                "lite_version": 1,\
                "model_path": "models/hopenet_lite_6MB.pkl",\
                "batch_size": 1,\
                "device": "cuda:0"\
            }\
        } \
    }\
    '
    task_dict=json.loads(json_data)
    
    # 创建并启动服务进程
    temp_input_q = mp.Queue(maxsize=10)
    temp_output_q = mp.Queue(maxsize=10)
    temp_process = mp.Process(target=work_func, args=("face_detection", temp_input_q, temp_output_q,
                                                      task_dict['model_ctx']["face_detection"]))
    temp_process.start()
    
    # 获取视频帧，构造输入交给服务进程执行
    video_path = "test-cut1.mp4"
    capture = cv2.VideoCapture(video_path)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    print("frame_height :{}, frame_width :{}".format(frame_height, frame_width))
    image = None
    
    if capture.isOpened():
        while True:
            ret,img=capture.read() # img 就是一帧图片            
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if ret:
                image = img
                break # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')
    
    expr_name = "face_detection_test_client_640_cpu"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    reso_list = ["360p", "720p", "1080p"]  # "360p", "480p", "720p", "1080p"
    for reso in reso_list:
        frame = cv2.resize(image, (
            resolution_wh[reso]['w'],
            resolution_wh[reso]['h']
        ))
        print("frame.shape is:{}".format(frame.shape))
        
        input_ctx = {'image': frame}
        temp_cpu_util_list = []
        temp_mem_util_list = []
        temp_latency_list = []
        temp_gpu_proc_time_list = []
        for i in range(100):
            temp_input_q.put(input_ctx)
            output_ctx = temp_output_q.get()
            temp_cpu_util_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
            temp_mem_util_list.append(output_ctx['proc_resource_info']['mem_util_use'])
            temp_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
            temp_gpu_proc_time_list.append(output_ctx['proc_resource_info']['gpu_proc_time'])
        
        temp_cpu_util_list = temp_cpu_util_list[1:]
        temp_cpu_util_row = [reso, 'cpu_util_use'] + temp_cpu_util_list
        data_list.append(temp_cpu_util_row)
        
        temp_mem_util_list = temp_mem_util_list[1:]
        temp_mem_util_row = [reso, 'mem_util_use'] + temp_mem_util_list
        data_list.append(temp_mem_util_row)
        
        temp_latency_list = temp_latency_list[1:]
        temp_latency_row = [reso, 'compute_latency'] + temp_latency_list
        data_list.append(temp_latency_row)
        
        temp_gpu_proc_time_list = temp_gpu_proc_time_list[1:]
        temp_gpu_proc_time_row = [reso, 'gpu_proc_time'] + temp_gpu_proc_time_list
        data_list.append(temp_gpu_proc_time_row)
        
        # print(temp_cpu_util_list, np.mean(temp_cpu_util_list))
        # print(temp_mem_util_list, np.mean(temp_mem_util_list))
        # print(temp_latency_list, np.mean(temp_latency_list))

    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)

        # 写入数据
        csv_writer.writerows(data_list)
'''



if __name__ == '__main__':
    # 测试单独在节点上运行服务时服务的执行情况（资源占用率、延时等），与服务运行在整个调度系统上时的执行情况是否一致
    # 此版本的主函数用于测试car_detection服务
    json_data=' \
    { \
        "name": "car_detection",  \
        "flow": ["car_detection"],\
        "model_ctx": {  \
            "car_detection": {\
                "weights": "yolov5l.pt",\
                "device": "cuda:0"\
            }\
        }  \
    }\
    '
    
    task_dict=json.loads(json_data)
    
    # 创建并启动服务进程
    temp_input_q = mp.Queue(maxsize=10)
    temp_output_q = mp.Queue(maxsize=10)
    temp_process = mp.Process(target=work_func, args=("car_detection", temp_input_q, temp_output_q,
                                                      task_dict['model_ctx']["car_detection"]))
    temp_process.start()
    
    # 获取视频帧，构造输入交给服务进程执行
    video_path = "test-cut2.mp4"
    capture = cv2.VideoCapture(video_path)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("frame_height :{}, frame_width :{}, fps :{}".format(frame_height, frame_width, fps))
    image = None
    
    if capture.isOpened():
        while True:
            ret,img=capture.read() # img 就是一帧图片            
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if ret:
                image = img
                break # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')
    
    expr_name = "car_detection_test_server_gpu"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    reso_list = ["360p", "480p", "720p", "1080p"]  # "360p", "480p", "720p", "1080p"
    for reso in reso_list:
        frame = cv2.resize(image, (
            resolution_wh[reso]['w'],
            resolution_wh[reso]['h']
        ))
        print("frame.shape is:{}".format(frame.shape))
        
        input_ctx = {'image': frame}
        temp_cpu_util_list = []
        temp_mem_util_list = []
        temp_mem_amount_list = []
        temp_latency_list = []
        temp_gpu_proc_time_list = []
        temp_pre_proc_time_list = []
        temp_post_proc_time_list = []
        for i in range(20):
            temp_input_q.put(input_ctx)
            output_ctx = temp_output_q.get()
            print(output_ctx['count_result'])
            temp_cpu_util_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
            temp_mem_util_list.append(output_ctx['proc_resource_info']['mem_util_use'])
            temp_mem_amount_list.append(output_ctx['proc_resource_info']['mem_use_amount'])
            temp_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
            temp_gpu_proc_time_list.append(output_ctx['proc_resource_info']['gpu_proc_time'])
            temp_pre_proc_time_list.append(output_ctx['proc_resource_info']['pre_proc_time'])
            temp_post_proc_time_list.append(output_ctx['proc_resource_info']['post_proc_time'])
        
        temp_cpu_util_list = temp_cpu_util_list[1:]
        temp_cpu_util_row = [reso, 'cpu_util_use'] + temp_cpu_util_list
        data_list.append(temp_cpu_util_row)
        
        temp_mem_util_list = temp_mem_util_list[1:]
        temp_mem_util_row = [reso, 'mem_util_use'] + temp_mem_util_list
        data_list.append(temp_mem_util_row)
        
        temp_mem_amount_list = temp_mem_amount_list[1:]
        temp_mem_amount_row = [reso, 'mem_use_amount'] + temp_mem_amount_list
        data_list.append(temp_mem_amount_row)
        
        temp_latency_list = temp_latency_list[1:]
        temp_latency_row = [reso, 'compute_latency'] + temp_latency_list
        data_list.append(temp_latency_row)
        
        temp_gpu_proc_time_list = temp_gpu_proc_time_list[1:]
        temp_gpu_proc_time_row = [reso, 'gpu_proc_time'] + temp_gpu_proc_time_list
        data_list.append(temp_gpu_proc_time_row)
        
        temp_pre_proc_time_list = temp_pre_proc_time_list[1:]
        temp_pre_proc_time_row = [reso, 'pre_proc_time'] + temp_pre_proc_time_list
        data_list.append(temp_pre_proc_time_row)
        
        temp_post_proc_time_list = temp_post_proc_time_list[1:]
        temp_post_proc_time_row = [reso, 'post_proc_time'] + temp_post_proc_time_list
        data_list.append(temp_post_proc_time_row)
        
        print(temp_pre_proc_time_list, np.mean(temp_pre_proc_time_list))
        print(temp_gpu_proc_time_list, np.mean(temp_gpu_proc_time_list))
        print(temp_post_proc_time_list, np.mean(temp_post_proc_time_list))
        print(np.mean(temp_gpu_proc_time_list) + np.mean(temp_post_proc_time_list))
        print(temp_latency_list, np.mean(temp_latency_list))
        # print(temp_mem_util_list, np.mean(temp_mem_util_list))
        # print(temp_latency_list, np.mean(temp_latency_list))

    # with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
    #     # 创建 CSV 写入对象
    #     csv_writer = csv.writer(csv_file)

    #     # 写入数据
    #     csv_writer.writerows(data_list)



'''
if __name__ == '__main__':
    # 创建并启动服务进程
    temp_input_q = mp.Queue(maxsize=10)
    temp_output_q = mp.Queue(maxsize=10)
    temp_process = mp.Process(target=work_func_test, args=(temp_input_q, temp_output_q))
    temp_process.start()
    
    temp_process_resource_limit = {
        'mem_util_limit': 1.0,
        'cpu_util_limit': 1.0
    }
    
    from cgroupspy import trees
    task_set = set()
    task_set.add(temp_process.pid)
    group_name = "process_" + str(temp_process.pid)
    t = trees.Tree()  # 实例化一个资源树

    # 限制进程使用的内存上限
    memory_resource_item = "memory"
    memory_limit_obj = t.get_node_by_path("/{0}/".format(memory_resource_item))
    memory_group = memory_limit_obj.create_cgroup(group_name)
    # 进程初始时设置内存上限为可使用全部内存
    memory_group.controller.limit_in_bytes = int(temp_process_resource_limit['mem_util_limit'] * psutil.virtual_memory().total)
    memory_group.controller.tasks = task_set
    

    # 限制进程的cpu使用率
    cpu_resource_item = "cpu"
    cpu_limit_obj = t.get_node_by_path("/{0}/".format(cpu_resource_item))
    cpu_group = cpu_limit_obj.create_cgroup(group_name)
    cpu_group.controller.cfs_period_us = 100000
    cpu_group.controller.cfs_quota_us = int(temp_process_resource_limit['cpu_util_limit'] * cpu_group.controller.cfs_period_us *
                                            psutil.cpu_count())
    cpu_group.controller.tasks = task_set
    
    input_ctx = 1
    temp_cpu_util_list = []
    temp_mem_util_list = []
    temp_compute_latency_list = []
    for i in range(50):
        temp_input_q.put(input_ctx)
        output_ctx = temp_output_q.get()
        temp_cpu_util_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
        temp_mem_util_list.append(output_ctx['proc_resource_info']['mem_util_use'])
        temp_compute_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
    print(temp_cpu_util_list, np.mean(temp_cpu_util_list[1:]))
    print(temp_mem_util_list, np.mean(temp_mem_util_list[1:]))
    print(temp_compute_latency_list, np.mean(temp_compute_latency_list[1:]))
'''

'''
if __name__ == '__main__':
    # 测试单独在节点上运行服务时服务的执行情况（资源占用率、延时等），与服务运行在整个调度系统上时的执行情况是否一致
    # 此版本的主函数用于测试face_detection服务
    json_data='\
    {\
        "name": "face_pose_estimation",  \
        "flow": ["face_detection", "face_alignment"],\
        "model_ctx": {  \
            "face_detection": {\
                "net_type": "mb_tiny_RFB_fd",\
                "input_size": 640,\
                "threshold": 0.7,\
                "candidate_size": 1500,\
                "device": "cpu"\
            },\
            "face_alignment": {\
                "lite_version": 1,\
                "model_path": "models/hopenet_lite_6MB.pkl",\
                "batch_size": 1,\
                "device": "cuda:0"\
            }\
        } \
    }\
    '
    task_dict=json.loads(json_data)
    
    # 创建并启动服务进程
    face_detection_input_q = mp.Queue(maxsize=10)
    face_detection_output_q = mp.Queue(maxsize=10)
    face_detection_process = mp.Process(target=work_func, args=("face_detection", face_detection_input_q, face_detection_output_q,
                                        task_dict['model_ctx']["face_detection"]))
    face_detection_process.start()
    
    
    face_alignment_input_q = mp.Queue(maxsize=10)
    face_alignment_output_q = mp.Queue(maxsize=10)
    face_alignment_process = mp.Process(target=work_func, args=("face_alignment", face_alignment_input_q, face_alignment_output_q,
                                        task_dict['model_ctx']["face_alignment"]))
    face_alignment_process.start()
    
    # 获取视频帧，构造输入交给服务进程执行
    video_path = "test-cut1.mp4"
    capture = cv2.VideoCapture(video_path)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    print("frame_height :{}, frame_width :{}".format(frame_height, frame_width))
    image = None
    
    if capture.isOpened():
        while True:
            ret,img=capture.read() # img 就是一帧图片            
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if ret:
                image = img
                break # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')
    
    expr_name = "face_alignment_test_client_640_gpu"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    reso_list = ["360p", "720p", "1080p"]  # "360p", "480p", "720p", "1080p"
    for reso in reso_list:
        frame = cv2.resize(image, (
            resolution_wh[reso]['w'],
            resolution_wh[reso]['h']
        ))
        print("frame.shape is:{}".format(frame.shape))
        
        input_ctx = {'image': frame}
        temp_cpu_util_list = []
        temp_mem_util_list = []
        temp_latency_list = []
        temp_gpu_proc_time_list = []
        for i in range(100):
            face_detection_input_q.put(input_ctx)
            output_ctx = face_detection_output_q.get()
            face_alignment_input_q.put(output_ctx)
            output_ctx_1 = face_alignment_output_q.get()
            
            temp_cpu_util_list.append(output_ctx_1['proc_resource_info']['cpu_util_use'])
            temp_mem_util_list.append(output_ctx_1['proc_resource_info']['mem_util_use'])
            temp_latency_list.append(output_ctx_1['proc_resource_info']['compute_latency'])
            temp_gpu_proc_time_list.append(output_ctx_1['proc_resource_info']['gpu_proc_time'])
        
        temp_cpu_util_list = temp_cpu_util_list[1:]
        temp_cpu_util_row = [reso, 'cpu_util_use'] + temp_cpu_util_list
        data_list.append(temp_cpu_util_row)
        
        temp_mem_util_list = temp_mem_util_list[1:]
        temp_mem_util_row = [reso, 'mem_util_use'] + temp_mem_util_list
        data_list.append(temp_mem_util_row)
        
        temp_latency_list = temp_latency_list[1:]
        temp_latency_row = [reso, 'compute_latency'] + temp_latency_list
        data_list.append(temp_latency_row)
        
        temp_gpu_proc_time_list = temp_gpu_proc_time_list[1:]
        temp_gpu_proc_time_row = [reso, 'gpu_proc_time'] + temp_gpu_proc_time_list
        data_list.append(temp_gpu_proc_time_row)
        
        # print(temp_cpu_util_list, np.mean(temp_cpu_util_list))
        # print(temp_mem_util_list, np.mean(temp_mem_util_list))
        # print(temp_latency_list, np.mean(temp_latency_list))

    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)

        # 写入数据
        csv_writer.writerows(data_list)
'''