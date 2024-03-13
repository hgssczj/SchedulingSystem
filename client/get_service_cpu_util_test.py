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
import math
from scipy.optimize import curve_fit
from sympy import symbols, diff


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


def work_func_test(q_input, q_output, sleep_second=0):
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
        
        end_cpu_per = p.cpu_percent(interval=None)  # 任务执行后进程的cpu占用率
        
        time.sleep(sleep_second)
        
        # res = 0
        # for i in range(1000000):
        #     res += 1

        # 延时的计算
        end_time = time.time()

        # cpu占用的计算
        # end_cpu_per = p.cpu_percent(interval=None)  # 任务执行后进程的cpu占用率
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


def get_service_compute_latency(q_input, q_output, cpu_group, memory_group, input_ctx, resource_limit_dict=None):
    '''
    获取某个服务在给定输入的情况下执行时的CPU利用率(中资源阈值)
    q_input: 服务获取输入的消息队列
    q_output: 服务放置输出的消息队列
    cpu_group: 控制服务cpu使用的cgroup
    memory_group: 控制服务mem使用的cgroup
    input_ctx: 服务的输入
    resource_limit_dict: 服务的资源限制方案
    '''
    if resource_limit_dict is None or ('cpu_util_limit' not in resource_limit_dict):
        cpu_util_limit = 1.0
    else:
        cpu_util_limit = resource_limit_dict['cpu_util_limit']
        
    if resource_limit_dict is None or ('mem_util_limit' not in resource_limit_dict):
        mem_util_limit = 1.0
    else:
        mem_util_limit = resource_limit_dict['mem_util_limit']
    
    # 根据资源限制方案修改服务进程的资源使用上限
    cpu_group.controller.cfs_quota_us = int(cpu_util_limit * cpu_group.controller.cfs_period_us * psutil.cpu_count())
    memory_group.controller.limit_in_bytes = int(mem_util_limit * psutil.virtual_memory().total)
    
    # 重复执行该服务若干次，将多次的时延求平均作为最终结果
    temp_compute_latency_list = []
    temp_cpu_util_use_list = []
    for i in range(5):
        q_input.put(input_ctx)
        output_ctx = q_output.get()
        temp_compute_latency_list.append(output_ctx['proc_resource_info']['compute_latency'])
        temp_cpu_util_use_list.append(output_ctx['proc_resource_info']['cpu_util_use'])
    latency = np.mean(temp_compute_latency_list[1:])  
    cpu_util_use = np.mean(temp_cpu_util_use_list[1:])
    
    return latency, cpu_util_use


def if_latency_deviation_large(temp_latency_list, latency_min, latency_dif):
    '''
    本函数用于判断时延列表temp_latency_list中是否大部分与latency_min的偏差都超出了latency_dif_ratio允许的范围
    temp_latency_list的长度必须是大于1的奇数
    '''
    output_info = []
    count = 0  # 记录temp_latency_list中时延偏差超出阈值的次数
    for temp_latency in temp_latency_list:
        if math.fabs(temp_latency - latency_min) >= latency_dif:
            count += 1
        output_info.append(math.fabs(temp_latency - latency_min))
    
    print("当前检验的CPU利用率其运行时延为:{}".format(temp_latency_list))
    print("当前检验的CPU利用率其运行时延与完美时延之间相差:{}".format(output_info))
    
    if count > len(temp_latency_list) - count:
        return True
    return False
    

def get_service_cpu_util_binary_search(proc, q_input, q_output, cpu_group, memory_group, input_ctx):
    '''
    本函数用于计算一次服务的cpu利用率
    '''
    ############################ 1.不限制CPU利用率，获取服务执行时延 ############################
    latency_min_list = []
    for i in range(3):  # 为了避免时延抖动带来的影响(尤其是边端)，计算三次时延，取最小的一次作为最终结果
        latency_min_list.append(get_service_compute_latency(q_input, q_output, cpu_group, memory_group, input_ctx))
    latency_min = np.min(latency_min_list) # 服务在不限制资源使用量情况下的时延
    
    # 根据latency_min的数量级确定可接受的相对于完美时延的偏差
    if latency_min < 0.01:
        latency_dif = 0.001
    elif latency_min < 0.03:
        latency_dif = 0.002
    elif latency_min < 0.05:
        latency_dif = 0.003
    elif latency_min < 0.075:
        latency_dif = 0.004
    elif latency_min < 0.1:  
        latency_dif = 0.005  # 小于100ms时cpu利用率的groundtruth很难测，而且此时边缘节点执行出现3ms左右的抖动很正常
    elif latency_min < 0.15:
        latency_dif = 0.007
    elif latency_min < 0.2:
        latency_dif = 0.007
    elif latency_min < 0.25:
        latency_dif = 0.007
    elif latency_min < 0.3:
        latency_dif = 0.015
    elif latency_min < 0.35:  
        latency_dif = 0.025
    elif latency_min < 0.40:
        latency_dif = 0.013
    elif latency_min < 0.45:
        latency_dif = 0.0265
    elif latency_min < 0.50:
        latency_dif = 0.015
    elif latency_min < 0.55:
        latency_dif = 0.024
    elif latency_min < 0.60:
        latency_dif = 0.015
    elif latency_min < 0.65:  # 这之前测的结果都比真实结果小0.01以内
        latency_dif = 0.0245
    elif latency_min < 0.70:  # 这种情况测出来的结果比真实结果小0.013左右
        latency_dif = 0.01
    elif latency_min < 0.75:  # 这种情况测出来的结果比真实结果小0.005到0.009左右
        latency_dif = 0.025
    elif latency_min < 0.80:  # 这种情况测出来的结果比真实结果小0.015左右
        latency_dif = 0.018 
    elif latency_min < 0.85:  # 这种情况测出来的结果比真实结果小0.011以内
        latency_dif = 0.029
    elif latency_min < 0.90:  # 这种情况测出来的结果比真实结果小0.004到0.017左右
        latency_dif = 0.015
    elif latency_min < 0.95:  # 这种情况测出来的结果比真实结果小0.01以内
        latency_dif = 0.025
    elif latency_min < 1.00:  # 这种情况测出来的结果比真实结果小0.02以内
        latency_dif = 0.015
    elif latency_min < 1.05:  # 这种情况测出来的结果比真实结果大0.004到0.01以内
        latency_dif = 0.025
    elif latency_min < 1.10:  # 这种情况测出来的结果比真实结果小0.011左右
        latency_dif = 0.020
    elif latency_min < 1.15:  # 这种测出来偏大0.01以内
        latency_dif = 0.030    
    elif latency_min < 1.20:  # 这种情况测出来的结果比真实结果小0.019左右
        latency_dif = 0.015
    elif latency_min < 1.25:  # 这种情况测出来的结果比真实结果小0.017左右
        latency_dif = 0.031
    elif latency_min < 1.30:  # 这种情况测出来的结果比真实结果小0.022左右
        latency_dif = 0.023
    elif latency_min < 1.35:  # 这种情况测出来的结果比真实结果大0.009以内左右
        latency_dif = 0.027
    elif latency_min < 1.40:  # 这种情况测出来的结果比真实结果小0.014以内左右
        latency_dif = 0.023
    elif latency_min < 1.45:  # 这种情况测出来的结果比真实结果大0.009以内左右
        latency_dif = 0.023
    elif latency_min < 1.50:  # 这种情况测出来的结果比真实结果小0.018以内左右
        latency_dif = 0.020
    else:
        latency_dif = 0.1  
    print("当前服务在资源不受限制的情况下,执行时延为: {}s, 对应的latency_dif为: {}s".format(latency_min, latency_dif))
    
    ############################ 2.直接使用小的delta，二分法确定中资源阈值 ############################
    delta_1 = 0.01  # 小的delta
    candidate_list_1 = [(i + 1) * delta_1 for i in range(int(1.0 / delta_1))]  # 确定大区间待查找的CPU利用率
    # print("第一轮大区间搜索,候选的CPU利用率为:{}\n".format(candidate_list_1))
    left_1 = 0
    right_1 = len(candidate_list_1) - 1
    res_1 = right_1
    while left_1 <= right_1:
        if left_1 < 0 or left_1 >= len(candidate_list_1):
            break
        if right_1 < 0 or right_1 >= len(candidate_list_1):
            break
        mid = (left_1 + right_1) // 2
        temp_cpu_util_limit = candidate_list_1[mid]  # 当前要验证的CPU利用率
        print("\n当前要检验的CPU利用率为:{}".format(temp_cpu_util_limit))
        temp_resource_limit_dict = {
            'cpu_util_limit': temp_cpu_util_limit,
            'mem_util_limit': 1.0
        }
        temp_latency_list = []  # 为了避免时延抖动带来的影响(尤其是边端)，计算三次时延
        for i in range(3):
            temp_latency_list.append(get_service_compute_latency(q_input, q_output, cpu_group, memory_group, input_ctx, temp_resource_limit_dict))
        
        # 取较多的结果作为最终结果
        if if_latency_deviation_large(temp_latency_list, latency_min, latency_dif):
            left_1 = mid + 1
        else:
            res_1 = min(res_1, mid)
            right_1 = mid - 1
    
    return candidate_list_1[res_1]
    # print("确定的大区间为:{}\n".format(candidate_list_1[res_1]))


def inv_prop_func(x, a, b, c):
    return a / (x ** b) + c


def get_service_cpu_util_curve_fit(proc, q_input, q_output, cpu_group, memory_group, input_ctx):
    '''
    本函数采用曲线拟合的方式测量cpu利用率
    '''
    cpu_util_2_latency = {}  # cpu利用率与时延之间的关系
    
    delta = 0.02  # 采样间隔
    temp_cpu_util_limit = delta  # [delta, 1.0]之间的利用率以delta为步长进行采样，不能采集利用率为0的情况
    while temp_cpu_util_limit <= 1.0:
        print("当前测试的CPU利用率为:{}".format(temp_cpu_util_limit))
        temp_resource_limit_dict = {
            'cpu_util_limit': temp_cpu_util_limit,
            'mem_util_limit': 1.0
        }
        temp_latency, _ = get_service_compute_latency(q_input, q_output, cpu_group, memory_group, input_ctx, temp_resource_limit_dict)
        cpu_util_2_latency[temp_cpu_util_limit] = temp_latency
        temp_cpu_util_limit += delta
    
    # 获取真实的CPU利用率
    _, real_cpu_util_use = get_service_compute_latency(q_input, q_output, cpu_group, memory_group, input_ctx)

    data_list = []
    cpu_util_list = []
    latency_list = []
    
    for cpu_util, latency in cpu_util_2_latency.items():
        temp_data_list = []
        temp_data_list.append(cpu_util)
        temp_data_list.append(latency)
        data_list.append(temp_data_list)
        
        cpu_util_list.append(cpu_util)
        latency_list.append(latency)
    
    # 拟合为反比例函数
    print("开始进行曲线拟合与求导")
    popt_1, pcov_1 = curve_fit(inv_prop_func, cpu_util_list, latency_list)
    
    # 计算在不限制CPU利用率时的时延
    latency_min = inv_prop_func(1.0, *popt_1)
    # 定义真实CPU利用率
    x_value = real_cpu_util_use
    # 计算在groundtruth利用率下的时延
    real_cpu_util_latency = inv_prop_func(x_value, *popt_1)
    # 计算二者的差，用于确定阈值
    latency_dif = real_cpu_util_latency - latency_min
    
    # 参数拟合的结果
    a_fit, b_fit, c_fit = popt_1
    # 定义符号变量
    x = symbols('x')
    # 定义拟合结果函数
    fit_function = a_fit / (x ** b_fit) + c_fit
    # 计算拟合函数关于 x 的导数
    derivative_fit_function = diff(fit_function, x)
    
    # 将 x 替换为指定值
    derivative_at_x = derivative_fit_function.subs(x, x_value)
    # 计算导数的数值
    derivative_value = derivative_at_x.evalf()
    
    expr_name = "test_process_cpu_util_{}".format(latency_min)  # 将当前采样得到的数据点输出到文件，便于后续出现问题进行分析
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
               '_' + expr_name + \
               '.csv'
    
    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)
        # 写入数据
        csv_writer.writerows(data_list)
    
    return latency_min, real_cpu_util_use, real_cpu_util_latency, latency_dif, derivative_value
            
        
def get_service_cpu_util(proc, q_input, q_output, cpu_group, memory_group, input_ctx, mode):
    '''
    本函数用于计算服务的CPU利用率
    '''
    # mode为0，则以二分查找的方式确定CPU利用率阈值。重复计算多次服务的cpu利用率, 观察cpu利用率的分布, 返回其均值或者分布特征
    if mode == 0:
        # ac_cpu_util_list = []
        # ac_latency_list = []
        # for i in range(10):
        #     q_input.put(input_ctx)
        #     temp_res = q_output.get()
        #     ac_cpu_util_list.append(temp_res['proc_resource_info']['cpu_util_use'])
        #     ac_latency_list.append(temp_res['proc_resource_info']['compute_latency'])
        
        # print("当前测试进程的实际CPU利用率为:{}, 均值为:{}\n".format(ac_cpu_util_list, np.mean(ac_cpu_util_list)))
        # print("当前测试进程的实际时延为:{}s, 均值为:{}s".format(ac_latency_list, np.mean(ac_latency_list)))
        
        num = 10
        cpu_util_list = []
        for i in range(num):
            print("\n##########开始第{}轮CPU利用率测量过程.##########".format(i))
            temp_cpu_util = get_service_cpu_util_binary_search(proc, q_input, q_output, cpu_group, memory_group, input_ctx)
            cpu_util_list.append(temp_cpu_util)

        cpu_util_mean = np.mean(cpu_util_list)
        cpu_util_var = np.var(cpu_util_list)
        cpu_util_std = np.std(cpu_util_list)
        
        print("{}次CPU利用率的均值为:{}, 方差为{}, 标准差为{}.".format(num, cpu_util_mean, cpu_util_var, cpu_util_std))
        print(cpu_util_list)
        return cpu_util_mean, cpu_util_var, cpu_util_std, cpu_util_list
    
    else:
        return get_service_cpu_util_curve_fit(proc, q_input, q_output, cpu_group, memory_group, input_ctx)


if __name__ == '__main__':
    # 以work_func_test作为模拟进程，用于确定方法的准确性
    sleep_time = 0.01
    data_list = []
    mode = 1
    while sleep_time < 1.5:
        print("当前测试的睡眠时间: {}s".format(sleep_time))
        # 创建并启动服务进程
        temp_input_q = mp.Queue(maxsize=10)
        temp_output_q = mp.Queue(maxsize=10)
        temp_process = mp.Process(target=work_func_test, args=(temp_input_q, temp_output_q, sleep_time))
        temp_process.start()
        
        temp_process_resource_limit = {
            'mem_util_limit': 1.0,
            'cpu_util_limit': 1.0
        }
        
        from cgroupspy import trees
        task_set = set()
        task_set.add(temp_process.pid)
        group_name = "process_" + str(temp_process.pid)
        print("当前进程的pid为:{}".format(temp_process.pid))
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
        
        
        if mode == 0:
            cpu_util_mean, cpu_util_var, cpu_util_std, cpu_util_list = get_service_cpu_util(temp_process, temp_input_q, temp_output_q, cpu_group, memory_group, input_ctx, mode)
        else:
            latency_min, temp_cpu_util_use, temp_latency, latency_dif, derivative_value = get_service_cpu_util(temp_process, temp_input_q, temp_output_q, cpu_group, memory_group, input_ctx, mode)
            temp_data_list = [sleep_time, latency_min, temp_cpu_util_use, temp_latency, latency_dif, derivative_value]
            data_list.append(temp_data_list)

        # 终止进程
        temp_process.terminate()
        # 等待进程结束
        temp_process.join()
        
        print("当前进程已被终止")
        sleep_time += 0.01
        time.sleep(3)  # 休息3s再进行下一轮循环

    if mode == 1:
        expr_name = "test_process_latency_2_threshold"
        filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
                '_' + expr_name + \
                '.csv'
        
        with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
            # 创建 CSV 写入对象
            csv_writer = csv.writer(csv_file)
            # 写入数据
            csv_writer.writerows(data_list)


'''
if __name__ == '__main__':
    # 测试单独在节点上运行服务时服务的CPU利用率
    # 此版本的主函数用于测试car_detection服务
    
    ################################ 1. 启动服务进程，并准备进程的输入数据 ################################
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
    
    ################################ 2. 在各种不同的配置下获取服务的CPU利用率 ################################
    # csv文件准备
    expr_name = "car_detection_test_client_gpu_cpu_util"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    # 遍历所有可能的分辨率，求出每种分辨率下的CPU利用率
    reso_list = ["360p", "480p", "720p", "1080p"]  # "360p", "480p", "720p", "1080p"
    for reso in reso_list:
        frame = cv2.resize(image, (
            resolution_wh[reso]['w'],
            resolution_wh[reso]['h']
        ))
        print("frame.shape is:{}".format(frame.shape))
        
        input_ctx = {'image': frame}
        cpu_util_mean, cpu_util_var, cpu_util_std, cpu_util_list = get_service_cpu_util(temp_process, temp_input_q, temp_output_q, cpu_group, memory_group, input_ctx)
        data_list.append([reso, cpu_util_mean, cpu_util_var, cpu_util_std] + cpu_util_list)
    
    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)
        # 写入数据
        csv_writer.writerows(data_list)
'''


'''
if __name__ == '__main__':
    # 测试单独在节点上运行服务时服务的CPU利用率
    # 此版本的主函数用于测试face_detection服务
    
    ################################ 1. 启动服务进程，并准备进程的输入数据 ################################
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
    
    # 获取视频帧，构造输入交给服务进程执行
    video_path = "test-cut1.mp4"
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
    
    ################################ 2. 在各种不同的配置下获取服务的CPU利用率 ################################
    # csv文件准备
    expr_name = "face_detection_test_client_gpu_cpu_util"
    filename = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + \
            '_' + expr_name + \
            '.csv'
    data_list = []
    
    # 遍历所有可能的分辨率，求出每种分辨率下的CPU利用率
    reso_list = ["360p", "480p", "720p", "1080p"]  # "360p", "480p", "720p", "1080p"
    for reso in reso_list:
        frame = cv2.resize(image, (
            resolution_wh[reso]['w'],
            resolution_wh[reso]['h']
        ))
        print("frame.shape is:{}".format(frame.shape))
        
        input_ctx = {'image': frame}
        cpu_util_mean, cpu_util_var, cpu_util_std, cpu_util_list = get_service_cpu_util(temp_process, temp_input_q, temp_output_q, cpu_group, memory_group, input_ctx)
        data_list.append([reso, cpu_util_mean, cpu_util_var, cpu_util_std] + cpu_util_list)
    
    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)
        # 写入数据
        csv_writer.writerows(data_list)
'''


'''
if __name__ == '__main__':
    # 测试单独在节点上运行服务时服务的执行情况（资源占用率、延时等），与服务运行在整个调度系统上时的执行情况是否一致
    # 此版本的主函数用于测试face_alignment服务
    ################################ 1. 启动服务进程，并准备进程的输入数据 ################################
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
    
    temp_process_resource_limit = {
        'mem_util_limit': 1.0,
        'cpu_util_limit': 1.0
    }
    
    from cgroupspy import trees
    task_set = set()
    task_set.add(face_alignment_process.pid)
    group_name = "process_" + str(face_alignment_process.pid)
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
    
    expr_name = "face_alignment_test_client_640_gpu_cpu_util"
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
        
        face_detection_input_q.put(input_ctx)
        output_ctx = face_detection_output_q.get()
        
        cpu_util_mean, cpu_util_var, cpu_util_std, cpu_util_list = get_service_cpu_util(face_alignment_process, face_alignment_input_q, face_alignment_output_q, cpu_group, memory_group, output_ctx)
        data_list.append([reso, cpu_util_mean, cpu_util_var, cpu_util_std] + cpu_util_list)
        

    with open(filename, 'w', newline='', encoding="utf-8") as csv_file:
        # 创建 CSV 写入对象
        csv_writer = csv.writer(csv_file)
        # 写入数据
        csv_writer.writerows(data_list)
'''