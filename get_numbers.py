﻿# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# 为了“开箱即用”，本脚本没有依赖除了Python库以外的组件。
# 添加自己的代码时，可以自由地引用如numpy这样的组件以方便编程。

import sys
import itertools
import numpy as np
import random

P = 0.3 #扰动几率
disturb_base = 75
disturb_range = 10
threhold = 0.1
small = 3

def LineToNums(line, type=float):
    """将输入的行按间隔符分割，并转换为某种数字类型的可迭代对象（默认为float类型）"""
    return (type(cell) for cell in line.split('\t'))

metaLine = sys.stdin.readline()
lineNum, columnNum = LineToNums(metaLine, int)
playersNum = (columnNum-1)//2

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

history = []
for line in map(lambda _: sys.stdin.readline(), range(lineNum)):
    gnum, *nums = LineToNums(line)
    history.append((gnum, nums))

def Mean(iter, len):
    """用于计算均值的帮主函数"""
    return sum(iter) / len

def Mean_candidiate(history, len):
    """用于计算均值的帮主函数"""
    h = history[-1 * len:]
    affix = [0.1, 0.1, 0.1, 0.2, 0.5]
    summ = 0
    for i in range(len):
        summ += affix[i] * history[i][0]
    return summ

#大数定理-返回预测的每个人下一轮的扰动值增加数值(不考虑自爆式两个扰动值的情况)
def pre_disturb_majority_rule(history, radios = 3):
    last_scores = history[-radios:]
    prediction_result = []
    flag = 0.95
    for i in range(playersNum):
            prediction_table = []
            for score in last_scores:
                if abs(score[1][2*i] - score[0]) > 20:
                    prediction_table.append(score[1][2*i] - score[0])
                elif abs(score[1][2*i+1] - score[0]) > 20:
                    prediction_table.append(score[1][2*i+1] - score[0])
            if len(prediction_table) > 0:
                prediction_result.append(Mean(prediction_table, len(prediction_table)))
                flag = 1.05
            else:
                prediction_result.append(0)
    #print(prediction_result)
    return prediction_result, flag

#概率期望-返回预测的每个人下一轮的扰动值增加数值(不考虑自爆式两个扰动值的情况)
def pre_disturb_probability(history, radios = 10):
    flag = 0.95
    last_scores = history[-radios:]
    prediction_result = []
    for i in range(playersNum):
            prediction_table = []
            count = 0
            for score in last_scores:
                if abs(score[1][2*i] - score[0]) > 20:
                    prediction_table.append(score[1][2*i] - score[0])
                    count = 1
                    flag = 1.05
                elif abs(score[1][2*i+1] - score[0]) > 20:
                    count = 1
                    flag = 1.05
                    prediction_table.append(score[1][2*i] - score[0])
                if count == 0:
                    prediction_table.append(0)
            prediction_result.append(Mean(prediction_table, len(prediction_table)))
    #print(prediction_result)
    return prediction_result, flag

if len(history) == 0:
    #disturb_table = np.zeros(int((columnNum - 1) / 2))
    #np.save('disturb_table.npy', disturb_table)
    prediction_table = np.zeros((2, 15))
    prediction_table[1][3] = 1
    prediction_table[0][0] = -1
    np.save('prediction_table.npy', prediction_table)
    prediction_result = np.zeros((2, 15))
    np.save('prediction_result.npy', prediction_result)
    print("17\t9.2")
else:
    # 取最近的记录，最多五项。计算这些记录中黄金点的均值作为本脚本的输出。
    #candidate1 = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5))
    #candidate2 = candidate1 * 0.618 # 第二输出。
    candidate_mean = Mean_candidiate(history, min(len(history), 5))
    candidate_mean_618 = 0.618 * candidate_mean
    candidate_618 = history[-1][0] * 0.618
    candidate_last_one = history[-1][0]
    x = np.array([1,2,3,4])
    if len(history) > 3:
        y = np.array([history[-4][0], history[-3][0], history[-3][0], history[-1][0]])
        last4 = y
        log_fit = np.polyfit(np.log(x),y,1)
        candidate_log = log_fit[0] * np.log(5) + log_fit[1]
    if len(history) < 4:
        print("%f\t%f" % (candidate_618, candidate_618+0.1*(random.random()*2-1)*candidate_618))
        exit(0)
    elif len(history) < 15:
        pre1, flag1 = pre_disturb_majority_rule(history, radios=3)
        pre2, flag2 = pre_disturb_probability(history, radios=len(history))
        
        # others_sum1 = candidate_mean * 2 * (playersNum) - sum(pre1)
        # others_sum1 = flag1 * others_sum1
        # others_sum2 = candidate_mean * 2 * (playersNum) - sum(pre2)
        # others_sum2 = flag2 * others_sum2
        #others_sum1 = others_sum2 = candidate_mean*2 * (playersNum-1)
        if random.random() < P and (is_smooth(last4) or small_enough(last4)):
            if candidate_mean > 35:
                if random.random() > 0.5:
                    disturb = -20 - 10 * random.random()
                else:
                    disturb = disturb_base + 0.1 * random.random()
                    if disturb + candidate_mean > 100:
                        disturb = 98 - candidate_mean
                # others_sum1 += disturb + candidate_last_one
                # others_sum2 += disturb + candidate_last_one
                # gama = 0.618/(2*playersNum+2)
                # candidate1 = gama * others_sum1 / (1 - gama)
                # candidate2 = gama * others_sum2 / (1 - gama)
                # candidate1 = others_sum1 / (2 * playersNum) * 0.618
                # candidate2 = others_sum2 / (2 * playersNum) * 0.618
                candidate1 = candidate_last_one + (disturb) / (2 * playersNum) * 0.618
                candidate2 = candidate_last_one + (disturb) / (2 * playersNum) * 0.618 
                if candidate1<=0 or candidate1>=100:
                    candidate1 = candidate_mean
                if candidate2<=0 or candidate1>=100:
                    candidate2 = candidate_mean
                if random.random() > 0.5:
                    print("%f\t%f" % (candidate1, disturb))
                else:
                    print("%f\t%f" % (candidate2, disturb))
                exit(0)
            else:
                disturb = disturb_base + disturb_range * random.random()
                if disturb + candidate_mean > 100:
                    disturb = 98 - candidate_mean
            # others_sum1 += disturb + candidate_mean
            # others_sum2 += disturb + candidate_mean
            # gama = 0.618 / (2 * playersNum + 2)
            # candidate1 = gama * others_sum1 / (1 - gama)
            # candidate2 = gama * others_sum2 / (1 - gama)
            # candidate1 = others_sum1 / (2 * playersNum) * 0.618
            # candidate2 = others_sum2 / (2 * playersNum) * 0.618
            candidate1 = candidate_last_one + (disturb) / (2 * playersNum) * 0.618
            candidate2 = candidate_last_one + (disturb) / (2 * playersNum) * 0.618 
            if candidate1 <= 0 or candidate1 >= 100:
                candidate1 = candidate_mean
            if candidate2 <= 0 or candidate1 >= 100:
                candidate2 = candidate_mean
            if random.random() > 0.5:
                print("%f\t%f" % (candidate1, disturb))
            else:
                print("%f\t%f" % (candidate2, disturb))
            exit(0)
        else:
            print("%f\t%f" % (candidate_mean_618, candidate_mean + 0.1 * (random.random() * 2 - 1) * candidate_618))
            exit(0)
    prediction_result = np.load('prediction_result.npy')
    prediction_table = np.load('prediction_table.npy')
    best_stratogy = 0
    best_radios = 3
    best_count = 0
    best_error = 100
    for i in range(2):
        for j in range(15):
            if(abs(prediction_result[i][j] - candidate_last_one) < best_error):
                best_error = abs(prediction_result[i][j] - candidate_last_one)
                best_stratogy = i
                best_radios = j
    prediction_table[best_stratogy][best_radios] += 1
    #print('result:')
    #print(prediction_result)
    #print(best_stratogy,best_radios,best_error)
    for i in range(2):
        for j in range(15):
            if prediction_table[i][j] >= best_count:
                best_stratogy = i
                best_radios = j
                best_count = prediction_table[i][j]
    np.save('prediction_table.npy',prediction_table)
    #print('table:')
    #print(prediction_table)
    #print(best_stratogy,best_radios,best_count)
    for radios in range(15):
        pre1, flag1 = pre_disturb_majority_rule(history, radios+1)
        pre2, flag2 = pre_disturb_probability(history, radios+1)
        # others_sum1 = candidate_mean * 2 * (playersNum - 1) - sum(pre1)
        # others_sum1 = flag1 * others_sum1
        # others_sum2 = candidate_mean * 2 * (playersNum - 1) - sum(pre2)
        # others_sum2 = flag2 * others_sum2
        #others_sum1 = others_sum2 = candidate_mean * 2 * (playersNum - 1)
        if True:
            if candidate_last_one > 35:
                if random.random() > 0.5:
                    disturb = -20 - 10 * random.random()
                else:
                    disturb = disturb_base + disturb_range * random.random()
                    if disturb + candidate_mean > 100:
                        disturb = 98 - candidate_mean
                # others_sum1 += disturb + candidate_mean
                # others_sum2 += disturb + candidate_mean
                # gama = 0.618 / (2 * playersNum + 2)
                # candidate1 = gama * others_sum1 / (1 - gama)
                # candidate2 = gama * others_sum2 / (1 - gama)
                # candidate1 = others_sum1 / (2 * playersNum) * 0.618
                # candidate2 = others_sum2 / (2 * playersNum) * 0.618
                candidate1 = candidate_mean + (disturb + candidate_mean) / (2 * playersNum) * 0.618
                candidate2 = candidate_mean + (disturb + candidate_mean) / (2 * playersNum) * 0.618 
                if candidate1 <= 0 or candidate1 >= 100:
                    candidate1 = candidate_last_one
                if candidate2 <= 0 or candidate1 >= 100:
                    candidate2 = candidate_last_one
                prediction_result[0][radios] = candidate1
                prediction_result[1][radios] = candidate2
                if best_radios == radios and best_stratogy==0:
                    if random.random() < P and (is_smooth(last4) or small_enough(last4)):
                        print("%f\t%f" % (candidate1, candidate_mean + disturb))
                    else:
                        print("%f\t%f" % (candidate_mean, candidate_mean + 0.1 * (random.random() * 2 - 1) * candidate_mean))
                elif best_radios == radios and best_stratogy==1:
                    if random.random() < P and (is_smooth(last4) or small_enough(last4)):
                        print("%f\t%f" % (candidate2, candidate_mean + disturb))
                    else:
                        print("%f\t%f" % (candidate_mean, candidate_mean + 0.1 * (random.random() * 2 - 1) * candidate_mean))
            else:
                disturb = disturb_base + disturb_range * random.random()
                if disturb + candidate_mean > 100:
                    disturb = 98 - candidate_mean
            # others_sum1 += disturb + candidate_mean
            # others_sum2 += disturb + candidate_mean
            # gama = 0.618 / (2 * playersNum + 2)
            # candidate1 = gama * others_sum1 / (1 - gama)
            # candidate2 = gama * others_sum2 / (1 - gama)
            # candidate1 = others_sum1 / (2 * playersNum) * 0.618
            # candidate2 = others_sum2 / (2 * playersNum) * 0.618
            candidate1 = candidate_last_one + (disturb) / (2 * playersNum) * 0.618
            candidate2 = candidate_last_one + (disturb) / (2 * playersNum) * 0.618 
            if candidate1 <= 0 or candidate1 >= 100:
                candidate1 = candidate_log
            if candidate2 <= 0 or candidate1 >= 100:
                candidate2 = candidate_log
            prediction_result[0][radios] = candidate1
            prediction_result[1][radios] = candidate2
            if best_radios == radios and best_stratogy == 0:
                if random.random() < P and (is_smooth(last4) or small_enough(last4)):
                    print("%f\t%f" % (candidate1, disturb))
                else:
                    print("%f\t%f" % (candidate_mean, candidate_last_one + 0.1 * (random.random() * 2 - 1) * candidate_mean))
            elif best_radios == radios and best_stratogy == 1:
                if random.random() < P and (is_smooth(last4) or small_enough(last4)):
                    print("%f\t%f" % (candidate2, disturb))
                else:
                    print("%f\t%f" % (candidate_mean, candidate_last_one + 0.1 * (random.random() * 2 - 1) * candidate_mean))

    np.save('prediction_result.npy', prediction_result)

    