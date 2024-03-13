# 本文件用于验证服务的CPU中资源阈值和内存中资源阈值是否相互独立

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

        # 延时的计算
        start_time = time.time()  # 计时，统计延时

        # 执行任务
        output_ctx = work_obj(input_ctx)

        # 延时的计算
        end_time = time.time()

        # cpu占用的计算
        end_cpu_per = p.cpu_percent(interval=None)  # 任务执行后进程的cpu占用率
    
        proc_resource_info = dict()  # 执行任务过程中的资源消耗情况
        proc_resource_info['pid'] = pid
        # 任务执行的cpu占用率，各个核上占用率之和的百分比->平均每个核的占用率，范围[0,1]
        proc_resource_info['cpu_util_use'] = end_cpu_per / 100 / psutil.cpu_count()
        proc_resource_info['mem_util_use'] = p.memory_percent(memtype='uss') / 100
        proc_resource_info['mem_use_amount'] = p.memory_full_info().uss
        proc_resource_info['compute_latency'] = end_time - start_time  # 任务执行的延时，单位：秒(s)

        output_ctx['proc_resource_info'] = proc_resource_info
        q_output.put(output_ctx)

'''
if __name__ == '__main__':
    # 创建并启动服务进程
    temp_input_q = mp.Queue(maxsize=10)
    temp_output_q = mp.Queue(maxsize=10)
    temp_process = mp.Process(target=work_func, args=("car_detection", temp_input_q, temp_output_q))
    temp_process.start()
    
    # 获取视频帧，构造输入交给服务进程执行
    expr_name = "change_mem_threshold_observe_cpu_util"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    frame = cv2.imread("1.jpg")
    input_ctx = {'image': frame}
    
    from cgroupspy import trees
    task_set = set()
    task_set.add(temp_process.pid)
    group_name = "process_" + str(temp_process.pid)
    t = trees.Tree()  # 实例化一个资源树
    
    # 限制进程使用的内存上限
    memory_resource_item = "memory"
    memory_limit_obj = t.get_node_by_path("/{0}/".format(memory_resource_item))
    memory_group = memory_limit_obj.create_cgroup(group_name)
    memory_group.controller.tasks = task_set
    mem_util_limit_list = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
    
    for temp_mem_limit in mem_util_limit_list:
        # 进程初始时设置内存上限为可使用全部内存
        memory_group.controller.limit_in_bytes = int(temp_mem_limit * psutil.virtual_memory().total)
        print("当前限制内存占用率为:{}, 占用内存值为:{}".format(temp_mem_limit, memory_group.controller.limit_in_bytes))

        temp_cpu_util_list = []
        temp_mem_util_list = []
        temp_mem_amount_list = []
        temp_latency_list = []
        for i in range(20):
            temp_input_q.put(input_ctx)
            output_ctx = temp_output_q.get()
            temp_cpu_util_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
            temp_mem_util_list.append(output_ctx['proc_resource_info']['mem_util_use'])
            temp_mem_amount_list.append(output_ctx['proc_resource_info']['mem_use_amount'])
            temp_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
        
        temp_cpu_util_list = temp_cpu_util_list[1:]
        
        temp_mem_util_list = temp_mem_util_list[1:]
        
        temp_mem_amount_list = temp_mem_amount_list[1:]
        
        temp_latency_list = temp_latency_list[1:]
        
        # print(temp_cpu_util_list, np.mean(temp_cpu_util_list))
        # print(temp_mem_util_list, np.mean(temp_mem_util_list))
        # print(temp_mem_amount_list, np.mean(temp_mem_amount_list))
        # print(temp_latency_list, np.mean(temp_latency_list))
        
        data_list.append([temp_mem_limit, np.mean(temp_cpu_util_list), np.mean(temp_mem_util_list), np.mean(temp_mem_amount_list), np.mean(temp_latency_list)])

    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)

        # 写入数据
        csv_writer.writerows(data_list)
'''

'''
if __name__ == '__main__':
    cpu_util_limit_list = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
                           0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    
    # 获取视频帧，构造输入交给服务进程执行
    expr_name = "change_cpu_threshold_observe_mem_util"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    frame = cv2.imread("1.jpg")
    input_ctx = {'image': frame}
    
    for temp_cpu_util_limit in cpu_util_limit_list:
        # 创建并启动服务进程
        temp_input_q = mp.Queue(maxsize=10)
        temp_output_q = mp.Queue(maxsize=10)
        temp_process = mp.Process(target=work_func, args=("car_detection", temp_input_q, temp_output_q))
        temp_process.start()
    
        from cgroupspy import trees
        task_set = set()
        task_set.add(temp_process.pid)
        group_name = "process_" + str(temp_process.pid)
        t = trees.Tree()  # 实例化一个资源树
    
        cpu_resource_item = "cpu"
        cpu_limit_obj = t.get_node_by_path("/{0}/".format(cpu_resource_item))
        cpu_group = cpu_limit_obj.create_cgroup(group_name)
        cpu_group.controller.cfs_period_us = 100000
        cpu_group.controller.cfs_quota_us = int(temp_cpu_util_limit * cpu_group.controller.cfs_period_us *
                                                psutil.cpu_count())
        cpu_group.controller.tasks = task_set
    
        print("当前限制CPU占用率为:{}".format(temp_cpu_util_limit))

        temp_cpu_util_list = []
        temp_mem_util_list = []
        temp_mem_amount_list = []
        temp_latency_list = []
        for i in range(20):
            temp_input_q.put(input_ctx)
            output_ctx = temp_output_q.get()
            temp_cpu_util_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
            temp_mem_util_list.append(output_ctx['proc_resource_info']['mem_util_use'])
            temp_mem_amount_list.append(output_ctx['proc_resource_info']['mem_use_amount'])
            temp_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
        
        temp_process.terminate()
        temp_process.join()
        
        temp_cpu_util_list = temp_cpu_util_list[1:]
        
        temp_mem_util_list = temp_mem_util_list[1:]
        
        temp_mem_amount_list = temp_mem_amount_list[1:]
        
        temp_latency_list = temp_latency_list[1:]
        
        # print(temp_cpu_util_list, np.mean(temp_cpu_util_list))
        # print(temp_mem_util_list, np.mean(temp_mem_util_list))
        # print(temp_mem_amount_list, np.mean(temp_mem_amount_list))
        # print(temp_latency_list, np.mean(temp_latency_list))
        
        data_list.append([temp_cpu_util_limit, np.mean(temp_cpu_util_list), np.mean(temp_mem_util_list), np.mean(temp_mem_amount_list), np.mean(temp_latency_list)])
        time.sleep(5)
        

    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)

        # 写入数据
        csv_writer.writerows(data_list)
'''

'''
if __name__ == '__main__':
    # 创建并启动服务进程
    temp_input_q = mp.Queue(maxsize=10)
    temp_output_q = mp.Queue(maxsize=10)
    temp_process = mp.Process(target=work_func, args=("car_detection", temp_input_q, temp_output_q))
    temp_process.start()
    
    # 获取视频帧，构造输入交给服务进程执行
    expr_name = "change_mem_threshold_observe_cpu_util"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    frame = cv2.imread("1.jpg")
    input_ctx = {'image': frame}
    
    from cgroupspy import trees
    task_set = set()
    task_set.add(temp_process.pid)
    group_name = "process_" + str(temp_process.pid)
    t = trees.Tree()  # 实例化一个资源树
    
    # 限制进程使用的内存上限
    memory_resource_item = "memory"
    memory_limit_obj = t.get_node_by_path("/{0}/".format(memory_resource_item))
    memory_group = memory_limit_obj.create_cgroup(group_name)
    memory_group.controller.tasks = task_set
    memory_group.controller.limit_in_bytes = int(0.002 * psutil.virtual_memory().total)
    print("当前限制内存占用率为:{}, 占用内存值为:{}".format(0.002, memory_group.controller.limit_in_bytes))
    
    cpu_resource_item = "cpu"
    cpu_limit_obj = t.get_node_by_path("/{0}/".format(cpu_resource_item))
    cpu_group = cpu_limit_obj.create_cgroup(group_name)
    cpu_group.controller.cfs_period_us = 100000
    cpu_group.controller.cfs_quota_us = int(0.60 * cpu_group.controller.cfs_period_us *
                                            psutil.cpu_count())
    cpu_group.controller.tasks = task_set

    temp_cpu_util_list = []
    temp_mem_util_list = []
    temp_mem_amount_list = []
    temp_latency_list = []
    for i in range(20):
        temp_input_q.put(input_ctx)
        output_ctx = temp_output_q.get()
        temp_cpu_util_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
        temp_mem_util_list.append(output_ctx['proc_resource_info']['mem_util_use'])
        temp_mem_amount_list.append(output_ctx['proc_resource_info']['mem_use_amount'])
        temp_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
    
    temp_cpu_util_list = temp_cpu_util_list[1:]
    
    temp_mem_util_list = temp_mem_util_list[1:]
    
    temp_mem_amount_list = temp_mem_amount_list[1:]
    
    temp_latency_list = temp_latency_list[1:]
    
    print(temp_cpu_util_list, np.mean(temp_cpu_util_list))
    print(temp_mem_util_list, np.mean(temp_mem_util_list))
    print(temp_mem_amount_list, np.mean(temp_mem_amount_list))
    print(temp_latency_list, np.mean(temp_latency_list))
''' 


if __name__ == '__main__':
    # 创建并启动服务进程
    temp_input_q = mp.Queue(maxsize=10)
    temp_output_q = mp.Queue(maxsize=10)
    temp_process = mp.Process(target=work_func, args=("car_detection", temp_input_q, temp_output_q))
    temp_process.start()
    
    # 获取视频帧，构造输入交给服务进程执行
    expr_name = "change_mem_threshold_and_cpu_threshold"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    frame = cv2.imread("1.jpg")
    input_ctx = {'image': frame}
    
    from cgroupspy import trees
    task_set = set()
    task_set.add(temp_process.pid)
    group_name = "process_" + str(temp_process.pid)
    t = trees.Tree()  # 实例化一个资源树
    
    # 限制进程使用的内存上限
    memory_resource_item = "memory"
    memory_limit_obj = t.get_node_by_path("/{0}/".format(memory_resource_item))
    memory_group = memory_limit_obj.create_cgroup(group_name)
    memory_group.controller.tasks = task_set
    mem_util_limit_list = [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
    
    cpu_resource_item = "cpu"
    cpu_limit_obj = t.get_node_by_path("/{0}/".format(cpu_resource_item))
    cpu_group = cpu_limit_obj.create_cgroup(group_name)
    cpu_group.controller.cfs_period_us = 100000
    cpu_group.controller.tasks = task_set
    cpu_util_limit_list = [1.0, 0.95, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    
    for temp_mem_limit in mem_util_limit_list:
        # 进程初始时设置内存上限为可使用全部内存
        memory_group.controller.limit_in_bytes = int(temp_mem_limit * psutil.virtual_memory().total)

        for temp_cpu_util_limit in cpu_util_limit_list:
            cpu_group.controller.cfs_quota_us = int(temp_cpu_util_limit * cpu_group.controller.cfs_period_us *
                                                psutil.cpu_count())
            
            print("当前限制内存占用率为:{}, CPU利用率为:{}".format(temp_mem_limit, temp_cpu_util_limit))
        
            temp_cpu_util_list = []
            temp_mem_util_list = []
            temp_mem_amount_list = []
            temp_latency_list = []
            for i in range(20):
                temp_input_q.put(input_ctx)
                output_ctx = temp_output_q.get()
                temp_cpu_util_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
                temp_mem_util_list.append(output_ctx['proc_resource_info']['mem_util_use'])
                temp_mem_amount_list.append(output_ctx['proc_resource_info']['mem_use_amount'])
                temp_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
            
            temp_cpu_util_list = temp_cpu_util_list[1:]
            
            temp_mem_util_list = temp_mem_util_list[1:]
            
            temp_mem_amount_list = temp_mem_amount_list[1:]
            
            temp_latency_list = temp_latency_list[1:]
            
            # print(temp_cpu_util_list, np.mean(temp_cpu_util_list))
            # print(temp_mem_util_list, np.mean(temp_mem_util_list))
            # print(temp_mem_amount_list, np.mean(temp_mem_amount_list))
            # print(temp_latency_list, np.mean(temp_latency_list))
        
            data_list.append([temp_mem_limit, temp_cpu_util_limit, np.mean(temp_cpu_util_list), np.mean(temp_mem_util_list), np.mean(temp_mem_amount_list), np.mean(temp_latency_list)])

    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)

        # 写入数据
        csv_writer.writerows(data_list)




