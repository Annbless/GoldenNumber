# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# 为了“开箱即用”，本脚本没有依赖除了Python库以外的组件。
# 添加自己的代码时，可以自由地引用如numpy这样的组件以方便编程。

import sys
import itertools
from RL import DeepQNetwork
import numpy as np
import logging
import random

number = 5
action = 11

def LineToNums(line, type=float):
    """将输入的行按间隔符分割，并转换为某种数字类型的可迭代对象（默认为float类型）"""
    return (type(cell) for cell in line.split('\t'))

metaLine = sys.stdin.readline()
lineNum, columnNum = LineToNums(metaLine, int)

history = []
for line in map(lambda _: sys.stdin.readline(), range(lineNum)):
    gnum, *nums = LineToNums(line)
    history.append((gnum, nums))

threhold = 0.1
small = 3

def Mean(iter, len):
    """用于计算均值的帮主函数"""
    return sum(iter) / len

def is_smooth(last4):
    diff = [(last4[i] - last4[i + 1]) for i in range(3)]
    if diff[1] <= 0 and diff[2] <= 0:
        return True
    diff = [abs(diff[i]) for i in range(3)]
    if diff[0] <= 2 * threhold * last4[0] and diff[1] <= threhold * last4[1] and diff[2] <= threhold * last4[2]:
        return True
    return False

def small_enough(last4):
    diff = [last4[i] - small for i in range(4)]
    if diff[0] <= 0 and diff[1] <= 0 and diff[2] <= 0 and diff[3] <= 0:
        return True
    
    return False



if len(history) == 0:
    table = np.zeros(int((columnNum-1)/2))
    np.save('table.npy',table)
    print("17\t9.2")
else:
    table = np.load('table.npy')
    for (i, number) in zip(range(int(columnNum-1)),history[-1][1]):
        if(number >= 50):
            table[i//2] = 1
    np.save('table.npy',table)
    
    candidate_mean = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5))
    candidate_mean_618 = 0.618 * candidate_mean
    candidate_618 = history[-1][0] * 0.618
    candidate_last_one = history[-1][0]
    
    if len(history) < 4:
        print("%f\t%f" % (candidate_618, candidate_618+0.1*(random.random()*2-1)*candidate_618))
        exit(0)
    
    last4 = history[-4:][0]
    x = np.array([1,2,3,4])
    y = np.array([history[-4][0], history[-3][0], history[-3][0], history[-1][0]])
    last4 = y
    linear_fit = np.polyfit(x,y,1)
    linear = np.poly1d(linear_fit)
    candidate_linear = linear(5)
    log_fit = np.polyfit(np.log(x),y,1)
    candidate_log = log_fit[0] * np.log(5) + log_fit[1]

    if is_smooth(last4):
        print("%f\t%f" % (99, candidate_last_one + 1.27))
        exit(0)
    if small_enough(last4):
        print("%f\t%f" % (99, candidate_last_one + 1.27))
        exit(0)
    
    print("%f\t%f" % (candidate_mean, candidate_log))
