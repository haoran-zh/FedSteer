#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:31:45 2023

@author: msiew, haoranz5
"""

# mmm (diffmodels) noniid

# currently----
# noniid mcm
# iid mcf

# future----
# iid mcf mcf
# noniid mc mc mc
# non iid m c_c100 m c c_100
# non iid m c f, m c f
import numpy as np
import matplotlib.pyplot as plt
from parserplot import ParserArgs
import os
from plotAllocation import plot_allocation
from plotAllocation import simulate_allocation
from plotAllocation import tasklist2clientlist
from plotAllocation import simulate_map
import sys
import pickle


def AvgAcc1trial(exp_array):
    exp_array_avg = np.mean(exp_array, axis=1)
    return exp_array_avg


def MinAcc1trial(exp_array):
    min_array = np.min(exp_array, axis=1)
    return min_array

def cascadeAcc(exp_array):
    cascade = 1
    for i in range(exp_array.shape[1]):
        cascade *= exp_array[:, i]
    print(cascade.shape)
    return cascade

def diff1trial(exp_array):
    diff = np.max(exp_array, axis=1) - np.min(exp_array, axis=1)
    return diff


def var1trial(exp_array):
    var = np.var(exp_array, axis=1)
    return var



numRounds = 1500

def sort_files(files):
    def extract_numbers(file_name):
        parts = file_name.split('_')
        exp_number = int(parts[3].replace('exp', ''))
        algo_number = int(parts[4].replace('algo', '').split('.')[0])
        return algo_number, exp_number

    return sorted(files, key=extract_numbers)

def load_extra_folder(folder_dict, seed=11, header='localAcc_'):
    # each folder should only contain 1 algo result_old
    # get keys of the folder_dict, key is the folder name
    files = []
    for key in folder_dict:
        this_path = os.path.join('./result', key+str(seed))
        try:
            files += [os.path.join(this_path, f) for f in os.listdir(this_path) if f.startswith(header)]
        except:
            return np.array([])
    exp_list = []
    for f in files:
        t = np.load(f)
        t = np.where(t <= 0, 0, t)  # t shape: (task_num, numRounds)
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120

    return exp_array

# read all files
# find all files starting with mcf
# algo_name = ["bayesian", "alpha-fairness", "random", "round robin","optimal_sampling"]

def generate_task_list(data, replace=-1):
    global client_num
    taskarray = np.ones((len(data), client_num))*replace
    for i in range(len(data)):
        for key in data[i]:
            taskarray[i][key] = data[i][key]
    return taskarray.tolist()

def allocation(folder_dict, seed=11):
    files = []
    for key in folder_dict:
        this_path = os.path.join('./result', key + str(seed))
        try:
            files += [os.path.join(this_path, f) for f in os.listdir(this_path) if f.startswith("allocation_dict")]
        except:
            return 0, 0
    if len(files) == 0:
        return 0, 0
    file = files[0]
    # open pickle file
    with open(file, 'rb') as f:
        data = pickle.load(f) # data is a list (len: 1500). each element is a dict, dict[client_id] = task_id
    task_list = generate_task_list(data)
    # plot allocation
    plot_allocation(task_list, this_path, round_num=all_rounds, algo=next(iter(folder_dict.values())))
    taskarray = np.array(task_list) # round_num client_num
    # count client resources for each task in each round
    task_count_all = []
    for i in range(all_rounds):
        task_thisround = taskarray[i]
        client_count = []
        # for each task, count the number of clients for this task
        task_num = 3
        for j in range(task_num):
            client_count.append(np.sum(task_thisround == j))
        task_count_all.append(client_count)
    task_count_all = np.array(task_count_all) # shape: numRounds task_index
    # compute variance of client count for each task
    var_tasks = []
    task_count_all_copy = task_count_all.copy()
    for i in range(task_num):
        # remove the round with 0 clients
        client_count = task_count_all_copy[:, i]
        client_count = client_count[client_count != 0]
        # compute variance
        var_tasks.append(np.var(client_count))
    return np.mean(var_tasks), task_count_all


def plot_each_task(all_algorithm_curve, keys_list):
    all_algorithm_curve = np.array(all_algorithm_curve)
    algorithm_num, task_num, round_num = all_algorithm_curve.shape
    #plt.clf()
    fig_eachTask, axs = plt.subplots(1, task_num, figsize=(8 * task_num, 5))
    for i in range(task_num):
        ax = axs[i]
        # average over seed
        for j in range(algorithm_num):
            curve = all_algorithm_curve[j]
            task_curve = curve[i, :]
            ax.plot(task_curve, label=f"{keys_list[j]}")
            ax.legend()
            ax.set_xlabel('Round')
            ax.set_ylabel('Global acc')
            ax.set_title(f'Global acc for task {i}')
    fig_eachTask.savefig(finalPath+'/global_acc_each_task.png')


def plot_allocation_count(all_count_list, keys_list):
    # size: algorithm seed round_num task_num
    # only keep the first seed
    count_list = [algo[0] for algo in all_count_list]
    all_count_list = np.array(count_list)
    algorithm_num, round_num, task_num = all_count_list.shape
    fig_eachTask, axs = plt.subplots(1, task_num, figsize=(8 * task_num, 5))
    for i in range(task_num):
        ax = axs[i]
        # average over seed
        for j in range(algorithm_num):
            curve = all_count_list[j]
            task_curve = curve[:, i]
            ax.plot(task_curve[200:230], label=f"{keys_list[j]}")
            ax.legend()
            ax.set_xlabel('Round')
            ax.set_ylabel('Global acc')
            ax.set_title(f'Global acc for task {i}')
    fig_eachTask.savefig(finalPath+'/task_allocation.png')



alpha = 1.5
alpha_ms = 4
"""extra_folder = {
    f"2task_nnn_0.01u_a{alpha_ms}_": f"a{alpha_ms}",
    f"2task_nnn_0.01u_AS_clientfair_a{alpha}_": f"AS_CF_a{alpha}",
    f"2task_nnn_0.01u_AS_taskfair_a{alpha}_": f"AS_TF_a{alpha}",
    f"2task_nnn_0.01u_ms_a{alpha_ms}_": f"ms_a{alpha_ms}",
    f"2task_nnn_0.01u_msAS_a{alpha_ms}_": f"msAS_a{alpha_ms}",
    f"2task_nnn_0.01u_OS_clientfair_a{alpha}_": f"OS_CF_a{alpha}",
f"2task_nnn_0.01u_OS_taskfair_a{alpha}_": f"OS_TF_a{alpha}",
"2task_nnn_0.01u_random_": "random",
}
seed_list = [11, 12, 13]"""

"""extra_folder = {
    #"1task_nnn_u91c0.3_agg_": "test",
    #"1task_nnn_u91c0.3_AS_a1_": "AS_a1",
    #"1task_nnn_u91c0.3_OS_a1_": "OS_a1",
    #"1task_nnn_u91c0.3_AS_a3_": "AS_a3",
    #"1task_nnn_u91c0.3_OS_a3_": "OS_a3",
    "1task_nnn_u91c0.3_qFel_a3_": "qFel_a2",
    #"1task_nnn_u91c0.3_test_a3_": "test_a3",
    "1task_nnn_u91c0.3_testfixed_a3_": "our proposed",
    #"1task_nnn_u91c0.3_testfixed2_a3_": "testfixed2_a3",
    #"1task_nnn_u91c0.3_testEloss_a3_": "testEloss_a3",
    #"1task_nnn_u91c0.3_random_": "random"
    "1task_nnn_u0.918753_random_": "random"
}
tasknum=1
all_rounds = 1500
seed_list = [14,15,16,17]
finalPath = f'./result/1task_nnn_u91c0.3_random_14'"""

"""extra_folder = {
    "3task_nnn_0.01u_a2_": "a2",
    "3task_nnn_0.01u_AS_clientfair_a2_": "AS_CF_a2",
    "3task_nnn_0.01u_AS_taskfair_a2_": "AS_TF_a2",
    "3task_nnn_0.01u_ms_a4_": "ms_a4",
    "3task_nnn_0.01u_msAS_a4_": "msAS_a4",
    "3task_nnn_0.01u_OS_clientfair_a2_": "OS_CF_a2",
    "3task_nnn_0.01u_OS_taskfair_a2_": "OS_TF_a2",
    "3task_nnn_0.01u_random_": "random"
}
seed_list = [11, 12, 13, 14]"""


"""extra_folder = {
    "4task_nnnn_0.01u_a2_": "a2",
    "4task_nnnn_0.01u_AS_clientfair_a2_": "AS_CF_a2",
    "4task_nnnn_0.01u_AS_taskfair_a2_": "AS_TF_a2",
    "4task_nnnn_0.01u_ms_a4_": "ms_a4",
    "4task_nnnn_0.01u_msAS_a4_": "msAS_a4",
    "4task_nnnn_0.01u_OS_clientfair_a2_": "OS_CF_a2",
    "4task_nnnn_0.01u_OS_taskfair_a2_": "OS_TF_a2",
    "4task_nnnn_0.01u_random_": "random"
}
seed_list = [12, 13]"""

"""# no fairness results
u_value = 0.1
d_value = 1.0
a = 1
extra_folder = {
#f"3task_nnn_u{u_value}d{d_value}_a{a}_": f"a{a}",
##f"3task_nnn_u{u_value}d{d_value}_AS_clientfair_a{a}_": f"AS_CF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_AS_taskfair_a{a}_": f"AS_TF_a{a}",
f"3task_nnn_u{u_value}d{d_value}_OS_clientfair_a{a}_": f"OS_CF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_OS_taskfair_a{a}_": f"OS_TF_a{a}",
f"3task_nnn_u{u_value}d{d_value}_random_": "random"
}
seed_list = [11, 12, 13, 14]"""


"""u_value = 0.7
d_value = 0.3
c = 0.2
a = 2
ms_a = 4
tasknum=5
extra_folder = {
#f"3task_nnn_u{u_value}d{d_value}_a{a}_": f"a{a}",
#f"3task_nnn_u{u_value}d{d_value}_ms_a{ms_a}_": f"ms_a{ms_a}",
f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_msAS_a{ms_a}_": f"msAS_a{ms_a}",
#f"3task_nnn_u{u_value}d{d_value}_AS_clientfair_a{a}_": f"AS_CF_a{a}",
f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_AS_taskfair_a{a}_": f"AS_TF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_OS_clientfair_a{a}_": f"OS_CF_a{a}",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_OS_taskfair_a{a}_": f"OS_TF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_qFel_a{a}_": f"qFel_a{a}",
f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_random_": "random",
f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_test2_a2_": "test2_a2",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_GS_a{a}_": "Group sample",
}
all_rounds=800
seed_list = [14,15,16,17,18,19,20]
finalPath = f'./result/{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_random_14'"""

"""u_value = 0.4
c = 0.3
a = 2
a2 = 1
ms_a = 4
tasknum=1
all_rounds=600
extra_folder = {
#f"3task_nnn_u{u_value}d{d_value}_a{a}_": f"a{a}",
#f"3task_nnn_u{u_value}d{d_value}_ms_a{ms_a}_": f"ms_a{ms_a}",
#f"{tasknum}task_nnn_u{u_value}c{c}_msAS_a{ms_a}_": f"msAS_a{ms_a}",
#f"3task_nnn_u{u_value}d{d_value}_AS_clientfair_a{a}_": f"AS_CF_a{a}",
f"{tasknum}task_nnn_u{u_value}c{c}_AS_a{a2}_": f"AS_a{a2}",
f"{tasknum}task_nnn_u{u_value}c{c}_AS_a{a}_": f"AS_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_OS_clientfair_a{a}_": f"OS_CF_a{a}",
f"{tasknum}task_nnn_u{u_value}c{c}_OS_a{a2}_": f"OS_a{a2}",
f"{tasknum}task_nnn_u{u_value}c{c}_OS_a{a}_": f"OS_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_qFel_a{a}_": f"qFel_a{a}",
f"{tasknum}task_nnn_u{u_value}c{c}_random_": "random",
f"{tasknum}task_nnn_u{u_value}c{c}_testfixed2_a3_": "test2_a2",
}
seed_list = [15, 16, 17, 18, 19]
finalPath = f'./result/{tasknum}task_nnn_u{u_value}c{c}_random_15'"""

"""u_value = 0.4
c = 0.3
a = 2
a2 = 1
ms_a = 4
tasknum=3
all_rounds=500
extra_folder = {
f"3task_nnnnn_c0.15u0.0d0.5_msASSP_a4_": "msASSP_a4",
f"3task_nnnnn_c0.15u0.0d0.5_test2SP_a2_": "test2SP_a2",
    "3task_nnnnn_c0.15u0.0d0.5_randomSP_": "randomSP",
    "3task_nnnnn_c0.15u0.0d0.5_ASSP_taskfair_a1_": "ASSP_TF_a1",
    "3task_nnnnn_c0.15u0.0d0.5_ASSP_taskfair_a2_": "ASSP_TF_a2",
}
seed_list = [14, 15, 16, 21, 22, 23]
finalPath = './result/3task_nnnnn_c0.15u0.0d0.5_msASSP_a4_14'"""

"""u_value = 0.4
c = 0.3
a = 2
a2 = 1
ms_a = 4
tasknum=4
all_rounds=800
extra_folder = {
f"4task_nnnnn_c0.3u0.0d1.0_msAS_a4_": "msASSP_a4",
f"4task_nnnnn_c0.3u0.0d1.0_test2_a2_": "test2SP_a2",
    "4task_nnnnn_c0.3u0.0d1.0_random_": "randomSP",
    "4task_nnnnn_c0.3u0.0d1.0_AS_taskfair_a2_": "ASSP_TF_a2",
}
seed_list = [18, 19, 20, 21]
finalPath = './result/4task_nnnnn_c0.3u0.0d1.0_random_18'"""


"""u_value = 0.9
c = 0.3
a = 2
a2 = 1
ms_a = 4
tasknum=1
all_rounds=1500
extra_folder = {
f"1task_nnn_u{u_value}18753_qFel_a2_": "qFel_a2",
f"1task_nnn_u{u_value}18753_random_": "random",
f"1task_nnn_u{u_value}18753_testfixed_a3_": "testfixed_a2",
}
seed_list = [14,15,16,17]
finalPath = f'./result/1task_nnn_u{u_value}18753_random_14'"""



"""u_value = 0.0
c = 0.3
a = 2
a2 = 1
ms_a = 4
tasknum=3
all_rounds=800
extra_folder = {
f"3task_nnnnn_c0.1u0.0d1.0_GS_a2_": "GS",
f"3task_nnnnn_c0.1u0.0d1.0_random_": "random",
f"3task_nnnnn_c0.1u0.0d1.0_AS_taskfair_a2_": "AS_TF_a2",
}
seed_list = [18,19,20]
finalPath = f'./result/3task_nnnnn_c0.1u0.0d1.0_random_18'"""


"""u_value = 0.9
d_value = 0.3
c = 0.1
a = 2
ms_a = 4
tasknum=3
client_num=80
# task=3 ,client_num=80
extra_folder = {
#f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_msAS_a{ms_a}_": f"msAS_a{ms_a}",
#f"3task_nnn_u{u_value}d{d_value}_AS_clientfair_a{a}_": f"AS_CF_a{a}",
f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_AS_taskfair_a{a}_": f"AS_TF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_OS_clientfair_a{a}_": f"OS_CF_a{a}",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_OS_taskfair_a{a}_": f"OS_TF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_qFel_a{a}_": f"qFel_a{a}",
f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_random_": "random",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_test2_a2_": "test2_a2",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_GSo_a{a}_": "Group sample",
f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_tradeoff_a{a}_": "tradeoff_a2",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_tradeoff_a{ms_a}_": "tradeoff_notfair",
}
all_rounds=150
seed_list = [14,15,16,17,18]
# sd 21 is good,
# sd 19, 20 is bad,
finalPath = f'./result/{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_random_14'"""


u_value = 0.9
d_value = 0.3
c = 0.1  # active rate
a = 1 # alpha
#ms_a = 4
tasknum= 5 # task number
client_num=120 # client number
# task=3 ,client_num=80
extra_folder = {
#f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_msAS_a{ms_a}_": f"msAS_a{ms_a}",
#f"3task_nnn_u{u_value}d{d_value}_AS_clientfair_a{a}_": f"AS_CF_a{a}",
#f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_OS_a{a}_": f"OS-gn",
####f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_OS-sub_a{a}_": "OS-gn-sub",
#f"3task_nnn_u{u_value}d{d_value}_OS_clientfair_a{a}_": f"OS_CF_a{a}",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_OS_taskfair_a{a}_": f"OS_TF_a{a}",
#f"3task_nnn_u{u_value}d{d_value}_qFel_a{a}_": f"qFel_a{a}",
#f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_AS_a{a}_": "OS-loss",
####f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_AS-sub_a{a}_": "OS-loss-sub",
##f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_test2_a2_": "test2_a2",
#f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_tradeoff_a{a}_": "tradeoff_a2",
#f"{tasknum}task_nnnnn_c{c}u{u_value}d{d_value}_random_": "random-active0.1",
#f"{tasknum}task_nnnnn_c{0.2}u{u_value}d{d_value}_random_": "random-active0.2",
#f"{tasknum}task_nnnnn_c{0.3}u{u_value}d{d_value}_random_": "random-active0.3",
#f"{tasknum}task_nnnnn_c{0.4}u{u_value}d{d_value}_random_": "random-active0.4",
#f"{tasknum}task_nnnnn_c{0.5}u{u_value}d{d_value}_random_": "random-active0.5",
#f"{tasknum}task_nnnnn_fffse_lessVenn_AS_a1_": "AS",
#f"{tasknum}task_nnnnn_fffse_lessVenn_OS_a1_": "OS",
#f"{tasknum}task_nnnnn_fffse_lessVenn_random_": "random",
#f"{tasknum}task_nnnnn_distribution_lessVennr300_AS_a1_": "AS",
#f"{tasknum}task_nnnnn_distribution_lessVennc0.1_ASF0.2_a1_": "ASF0.2",
##f"{tasknum}task_nnnnn_distribution_lessVennc0.1_ASF0.1_a1_": "ASF0.1",
#f"{tasknum}task_nnnnn_distribution_lessVennc0.1_ASF0.05_a1_": "ASF0.05",
#f"{tasknum}task_nnnnn_distribution_lessVennc0.1_ASF0.01_a1_": "ASF0.01",
#f"{tasknum}task_nnnnn_distribution_lessVenn_ASF0.05_a1_": "ASF0.05",
#f"{tasknum}task_nnnnn_distribution_lessVenn_ASF0.01_a1_": "ASF0.01",
# f"{tasknum}task_nnnnn_distribution_lessVenn_OS_a1_": "OS",
# #f"{tasknum}task_nnnnn_distribution_lessVennr300_accAS_a1_": "AS-acc",
# f"{tasknum}task_nnnnn_distribution_lessVenn_AS_a1_": "AS",
# f"{tasknum}task_nnnnn_distribution_lessVennc0.1_full_": "full-participation",
# f"{tasknum}task_nnnnn_distribution_lessVenn_random_": "random"
######f"{tasknum}task_nnnnn_icdcs_c{c}u{u_value}d{d_value}_a1_": "a1",
######f"{tasknum}task_nnnnn_icdcs_c{c}u{u_value}d{d_value}_a2_": "a2",
 #"5task_nnnnn_fairness_ms_a2_": "ms_a2_data0.5",
#"5task_nnnnn_fairness_alphafair_a2_": "alphafair_a2_data0.5",
#"5task_nnnnn_fairness_random_": "random_data0.5",
# "3task_nnnnn_fairnessfff_ms_a1_": "ms_a1_data1.0",
# "3task_nnnnn_fairnessfff_alphafair_a1_": "alphafair_a1_data1.0",
# "3task_nnnnn_fairnessfff_random_": "random_data1.0",
# "3task_nnnnn_fairnessfff_AS_a1_": "AS_data1.0",

# "3task_nnnnn_fairnessfff_ms_a2c120_": "ms_a1_data1.0",
# "3task_nnnnn_fairnessfff_alphafair_a2c120_": "alphafair_a2_c120",
# "3task_nnnnn_fairnessfff_randomc120_": "random_data1.0",
# "3task_nnnnn_fairnessfff_AS_a1_": "AS_data1.0",

 #"3task_nnnnn_fairness_AS_a3_": "AS_data1.0",
#"5task_nnnnn_fairness_ms_a3_": "ms_a3_data1.0",
#"5task_nnnnn_fairness_alphafair_a3_": "alphafair_a3_data1.0",
#"5task_nnnnn_fairness_random_": "random_data1.0",
#"5task_nnnnn_fairness_AS_a3_": "AS_data1.0",

# f"{tasknum}task_nnnnn_fffse_lessVenn_OS_a1_": "OS",
# #f"{tasknum}task_nnnnn_distribution_lessVennr300_accAS_a1_": "AS-acc",
# f"{tasknum}task_nnnnn_fffse_lessVenn_AS_a1_": "AS",
# f"{tasknum}task_nnnnn_fffse_lessVenn_full_": "full-participation",
# f"{tasknum}task_nnnnn_fffse_lessVenn_random_": "random"

# f"{tasknum}task_nnnnn_mse_lessVennc0.1uv0.90.05_OS_a1_": "OS", # not use mse, just forget to change the name
# f"{tasknum}task_nnnnn_mse_lessVennc0.1uv0.90.05_AS_a1_": "AS", # more unbalanced case
# f"{tasknum}task_nnnnn_mse_lessVennc0.1uv0.90.05_full_": "full-participation",
# f"{tasknum}task_nnnnn_mse_lessVennc0.1uv0.90.05_random_": "random"


# f"{tasknum}task_nnnnn_distribution_lessVennssp_ASF0.1_a1_": "ASF0.1", # not use mse, just forget to change the name
# f"{tasknum}task_nnnnn_distribution_lessVennssp_ASF0.01_a1_": "ASF0.01",
# f"{tasknum}task_nnnnn_distribution_lessVennssp_ASF0.05_a1_": "ASF0.05",
# f"{tasknum}task_nnnnn_distribution_lessVennssp_AS_a1_": "AS"

# "1task_nnnnn_lessVennc0.1uv0.9_AS_a1_": "AS",
# "1task_nnnnn_lessVennc0.1uv0.9_OS_a1_": "OS",
# "1task_nnnnn_lessVennc0.1uv0.9_full_": "full participation",
# "1task_nnnnn_lessVennc0.1uv0.9_random_": "random",

# "5task_nnnnn_distribution_lessVenn_AS_a1c1.0u0.3_": "AS",
# "5task_nnnnn_distribution_lessVenn_ASF0.1_a1c1.0u0.3_": "ASF0.1",
# "5task_nnnnn_distribution_lessVenn_ASF0.05_a1c1.0u0.3_": "ASF0.05",
# "5task_nnnnn_distribution_lessVenn_ASF0.01_a1c1.0u0.3_": "ASF0.01",

# "5task_nnnnn_fairnessfff_ms_a2c120_": "MMFL-FairVR", # data ratio=0.3
# "5task_nnnnn_fairnessfff_alphafair_a2c120_": "FedFairMMFL",
# "5task_nnnnn_fairnessfff_randomc120_": "Random",

# "1task_nnnnn_lessVennc0.05uv0.9_AS_a1_": "AS", # active rate-0.05, uv 0.9 0.1, data ratio 1.0
# "1task_nnnnn_lessVennc0.05uv0.9_ASF0.0_": "ASF0.0",
# "1task_nnnnn_lessVennc0.05uv0.9_ASF0.01_": "ASF0.01",
# "1task_nnnnn_lessVennc0.05uv0.9_ASF0.05_": "ASF0.05",
# "1task_nnnnn_lessVennc0.05uv0.9_ASFslow0.01_": "ASFslow0.01",
# "1task_nnnnn_lessVennc0.05uv0.9_ASFslow0.05_": "ASFslow0.05",
# "1task_nnnnn_lessVennc0.05uv0.9_OS_a1_": "OS",
# "1task_nnnnn_lessVennc0.05uv0.9_OSF0.0_": "OSF0.0",
# "1task_nnnnn_lessVennc0.05uv0.9_OSF0.01_": "OSF0.01",
# "1task_nnnnn_lessVennc0.05uv0.9_OSF0.05_": "OSF0.05",
# "1task_nnnnn_lessVennc0.05uv0.9_OSFslow0.01_": "OSFslow0.01",
# "1task_nnnnn_lessVennc0.05uv0.9_OSFslow0.05_": "OSFslow0.05",
# "1task_nnnnn_lessVennc0.05uv0.9_full_": "full participation",
# "1task_nnnnn_lessVennc0.05uv0.9_random_": "random",


 # active rate-0.05, uv 0.9 0.1, data ratio 1.0
# "1task_nnnnn_lessVennc0.05uv0.90.01_ASF0.0_": "ASF0.0",
# "1task_nnnnn_lessVennc0.05uv0.90.01_ASF0.01_": "ASF0.01",
# "1task_nnnnn_lessVennc0.05uv0.90.01_ASF0.05_": "ASF0.05",
# "1task_nnnnn_lessVennc0.05uv0.90.01_AS_a1_": "AS",
# # "1task_nnnnn_lessVennc0.05uv0.90.01_ASFslow0.01_": "ASFslow0.01",
# # "1task_nnnnn_lessVennc0.05uv0.90.01_ASFslow0.05_": "ASFslow0.05",
# # "1task_nnnnn_lessVennc0.05uv0.90.01_OSF0.0_": "OSF0.0",
# # "1task_nnnnn_lessVennc0.05uv0.90.01_OSF0.01_": "OSF0.01",
# # "1task_nnnnn_lessVennc0.05uv0.90.01_OSF0.05_": "OSF0.05",
# "1task_nnnnn_lessVennc0.05uv0.90.01_OS_a1_": "OS",
# "1task_nnnnn_lessVennc0.05uv0.90.01_OSdelta0.001_": "OSdelta0.001",
# "1task_nnnnn_lessVennc0.05uv0.90.01_OSdelta0.005_": "OSdelta0.005",
# "1task_nnnnn_lessVennc0.05uv0.90.01_OSdelta0.01_": "OSdelta0.01",
# "1task_nnnnn_lessVennc0.05uv0.90.01_OSdelta0.02_": "OSdelta0.02",
# "1task_nnnnn_lessVennc0.05uv0.90.01_OSdelta0.03_": "OSdelta0.03",
# # "1task_nnnnn_lessVennc0.05uv0.90.01_OSFslow0.01_": "OSFslow0.01",
# # "1task_nnnnn_lessVennc0.05uv0.90.01_OSFslow0.05_": "OSFslow0.05",
# "1task_nnnnn_lessVennc0.05uv0.90.01_full_": "full participation",
# "1task_nnnnn_lessVennc0.05uv0.90.01_random_": "random",

# "5task_nnnnn_lessVennc0.1uv0.90.05_AS_": "AS", # active rate-0.1, uv 0.9 0.05, data ratio 0.5
# "5task_nnnnn_lessVennc0.1uv0.90.05_OS_": "OS",
# "5task_nnnnn_lessVennc0.1uv0.90.05_random_": "random",
# "5task_nnnnn_lessVennc0.1uv0.90.05_full_": "full participation",

 # active rate-0.05, uv 0.9 0.1, data ratio 1.0
# # infocom experiment, 3 tasks
# "3task_nnnnn_lessVennc0.1uv0.90.1_OS_": "MMFL-GVR",
# "3task_nnnnn_lessVennc0.1uv0.90.1_AS_": "MMFL-LVR",
# "3task_nnnnn_lessVennc0.1uv0.90.1_Ob_f_": "MMFL-StaleVR",
# "3task_nnnnn_lessVennc0.1uv0.90.1_GVRsequential_f_": "Sequential-SMVR",
# # # "3task_nnnnn_lessVennc0.1uv0.90.1_GVRsequential_icdcs_": "Sequential-SMVR2",
# "3task_nnnnn_lessVennc0.1uv0.90.1_FedVARP_a1_": "FedVARP",
# "3task_nnnnn_lessVennc0.1uv0.90.1_MILA_a1_": "MIFA",
# "3task_nnnnn_lessVennc0.1uv0.90.1_SCAFFOLD_a1_": "SCAFFOLD",
# "3task_nnnnn_lessVennc0.1uv0.90.1_full_ca_": "Full participation",
# "3task_nnnnn_lessVennc0.1uv0.90.1_random_ca_": "Random",


# infocom 5 tasks experiments
# "5task_nnnnn_lessVennc0.1uv0.60.1_OS_f2ca_": "MMFL-GVR",
# "5task_nnnnn_lessVennc0.1uv0.60.1_AS_f2ca_": "MMFL-LVR",
# "5task_nnnnn_lessVennc0.1uv0.60.1_Ob_f_": "MMFL-StaleVR",
# "5task_nnnnn_lessVennc0.1uv0.60.1_GVRsequential_f_": "Sequential-SMVR",
# "5task_nnnnn_lessVennc0.1uv0.60.1_FedVARP_f2_": "FedVARP",
# "5task_nnnnn_lessVennc0.1uv0.60.1_MILA_f2_": "MIFA",
# "5task_nnnnn_lessVennc0.1uv0.60.1_SCAFFOLD_f2_": "SCAFFOLD",
# "5task_nnnnn_lessVennc0.1uv0.60.1_full_f2ca_": "Full participation",
# "5task_nnnnn_lessVennc0.1uv0.60.1_random_f2ca_": "Random",

# icdcs 2 tasks, MMFL vs SMFLs
# "2task_nnnnn_lessVennc0.3uv0.90.1_GVR_icdcs_": "MMFL-GVR",
# "2task_nnnnn_lessVennc0.3uv0.90.1_GVRsequential_icdcs_": "Sequential-SMVR",

# "3task_nnnnn_lessVennc0.3uv0.90.1_GVR_icdcs_": "MMFL-GVR",
# "3task_nnnnn_lessVennc0.3uv0.90.1_GVRsequential_icdcs_": "Sequential-SMVR",

# today I needs to run these three algorithms
# optimal sampling with stale updates fixed beta: 0.5 0.6 0.7 0.8 0.9
# optimal sampling (Approx) with stale updates fixed beta: 0.5 0.6 0.7 0.8 0.9
# no optimal sampling with stale updates fixed beta: 0.5 0.6 0.7 0.8 0.9

# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_Ob_f_": "VR+optimal_b",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_decay_f_": "VRApprox+decayApprox",
# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_decay_f_": "VR+decayApprox",
# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_b0.7_f_": "VR+decay0.7",
# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_b0.8_f_": "VR+decay0.8",
# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_b0.9_f_": "VR+decay0.9",
# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_b1.0_f_": "VR+decay1.0",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_b0.7_f_": "VRApprox+decay0.7",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_b0.8_f_": "VRApprox+decay0.8",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_b0.9_f_": "VRApprox+decay0.9",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_b1.0_f_": "VRApprox+decay1.0",
# "1task_nnnnn_lessVennc0.1uv0.90.1_full_f_": "full participation",
# "1task_nnnnn_lessVennc0.1uv0.90.1_random_f_": "random",

# "1task_nnnnn_lessVennc0.03uv0.50.1_Diffstale_Ob_m_": "VR+optimal_b",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextra_decay_m_": "VRApprox+decayApprox",
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextraAO_decay_m_": "VRApprox+decayApprox",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_Diffstale_m_": "VR",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextra_m_": "VRApprox",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_Diffstale_decay_m_": "VR+decayApprox",
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextraw10_decay_m_": "VRApprox+decayApprox-w10",
# # should use AO
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextraw20_decay_m_": "VRApprox+decayApprox-w20",
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextraw30_decay_m_": "VRApprox+decayApprox-w30",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_Diffstale_b0.7_m_": "VR+decay0.7",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_Diffstale_b0.8_m_": "VR+decay0.8",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_Diffstale_b0.9_m_": "VR+decay0.9",
# #"1task_nnnnn_lessVennc0.03uv0.50.1_Diffstale_b1.0_m_": "VR+decay1.0",
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextra_b0.7_m_": "VRApprox+b0.7",
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextra_b0.8_m_": "VRApprox+b0.8",
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextra_b0.9_m_": "VRApprox+b0.9",
# "1task_nnnnn_lessVennc0.03uv0.50.1_DiffstaleNoextra_b1.0_m_": "VRApprox+b1.0",
# "1task_nnnnn_lessVennc0.03uv0.50.1_full_m_": "full participation",
# "1task_nnnnn_lessVennc0.03uv0.50.1_random_m_": "random",

# test all fashion-mnist, to show GVR code is ok
# "1task_nnnnn_lessVennc0.1uv0.90.1_GVR_f_": "GVR",
# "1task_nnnnn_lessVennc0.1uv0.90.1_GVRApprox_f_": "GVR_approx",
# "1task_nnnnn_lessVennc0.1uv0.90.1_full_f_": "full participation",
# "1task_nnnnn_lessVennc0.1uv0.90.1_random_f_": "random",


# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_Ob_m_": "VR+optimal_b",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_decay_m_": "VRApprox+decayApprox",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_decay_m2_": "VRApprox+decayApprox2",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_decay_m2_": "VRApprox+decayApprox",
# "1task_nnnnn_lessVennc0.1uv0.90.1_GVR_": "GVR", # need to check GVR code
# # "1task_nnnnn_lessVennc0.1uv0.90.1_GVRApprox_": "GVRApprox",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_decay_m_": "VRApproxAO+decayApprox",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw10_decay_m_": "VRApprox+decayApproxw10",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw5AO_decay_m2_": "VRApprox+decayApprox-w5",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw10AO_decay_m2_": "VRApprox+decayApprox-w10",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw15AO_decay_m2_": "VRApprox+decayApprox-w15", # use main2, not mnist2
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw20AO_decay_m2_": "VRApprox+decayApprox-w20",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw30AO_decay_m2_": "VRApprox+decayApprox-w30",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw40AO_decay_m2_": "VRApprox+decayApprox-w40",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw50AO_decay_m2_": "VRApprox+decayApprox-w50",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw70AO_decay_m2_": "VRApprox+decayApprox-w70",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw90AO_decay_m2_": "VRApprox+decayApprox-w90",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw120AO_decay_m2_": "VRApprox+decayApprox-w120",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW5AO_decay_m3_": "VRApprox+decayApprox-UW5",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW10AO_decay_m3_": "VRApprox+decayApprox-UW10",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW15AO_decay_m3_": "VRApprox+decayApprox-UW15", # use main2, not mnist2
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW20AO_decay_m3_": "VRApprox+decayApprox-UW20",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW30AO_decay_m3_": "VRApprox+decayApprox-UW30",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW40AO_decay_m3_": "VRApprox+decayApprox-UW40",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW90AO_decay_m3_": "VRApprox+decayApprox-UW90",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW120AO_decay_m3_": "VRApprox+decayApprox-UW120",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff5AO_decay_m3_": "VRApprox+decayApprox-UWff5",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff10AO_decay_m3_": "VRApprox+decayApprox-UWff10",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff15AO_decay_m3_": "VRApprox+decayApprox-UWff15", # use main2, not mnist2
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff20AO_decay_m3_": "VRApprox+decayApprox-UWff20",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff30AO_decay_m3_": "VRApprox+decayApprox-UWff30",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff40AO_decay_m3_": "VRApprox+decayApprox-UWff40",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff90AO_decay_m3_": "VRApprox+decayApprox-UWff90",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWff120AO_decay_m3_": "VRApprox+decayApprox-UWff120",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWam5AO_decay_m3_": "VRApprox+decayApprox-UWam5",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWam10AO_decay_m3_": "VRApprox+decayApprox-UWam10",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWam15AO_decay_m3_": "VRApprox+decayApprox-UWam15", # use main2, not mnist2
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWam20AO_decay_m3_": "VRApprox+decayApprox-UWam20",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWam30AO_decay_m3_": "VRApprox+decayApprox-UWam30",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUWam40AO_decay_m3_": "VRApprox+decayApprox-UWam40",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw30_decay_m_": "VRApprox+decayApproxw30",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_b0.9_m_": "VRApprox+b0.9",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.1_m2_": "VRApprox+b0.1",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.2_m2_": "VRApprox+b0.2",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.3_m2_": "VRApprox+b0.3",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.4_m2_": "VRApprox+b0.4",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.5_m2_": "VRApprox+b0.5",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.6_m2_": "VRApprox+b0.6",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.7_m2_": "VRApprox+b0.7",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.8_m2_": "VRApprox+b0.8",
# # "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.9_m2_": "VRApprox+b0.9",
# "1task_nnnnn_lessVennc0.1uv0.90.1_full_m_": "full participation",
# "1task_nnnnn_lessVennc0.1uv0.90.1_random_m_": "random",




# "3task_nnnnn_lessVennc0.3uv0.90.1_Diffstale_Ob_mfe2_": "VR+optimal_b",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_decay_m_": "VRApprox+decayApprox",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_decay_m2_": "VRApprox+decayApprox2",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraAO_decay_mfe2_": "VRApprox+decayApprox",
# # "3task_nnnnn_lessVennc0.3uv0.90.1_GVR_mfe_": "GVR",  # need to check GVR code
# # "3task_nnnnn_lessVennc0.3uv0.90.1_GVRApprox_mfe_": "GVRApprox",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_decay_m_": "VRApproxAO+decayApprox",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw10_decay_m_": "VRApprox+decayApproxw10",
# # "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraw5AO_decay_mfe2_": "VRApprox+decayApprox-w5",
# # "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraw10AO_decay_mfe2_": "VRApprox+decayApprox-w10",
# # "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraw15AO_decay_mfe2_": "VRApprox+decayApprox-w15",
# # "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraw20AO_decay_mfe2_": "VRApprox+decayApprox-w20",
# # "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraw30AO_decay_mfe2_": "VRApprox+decayApprox-w30",
# # "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraw40AO_decay_mfe2_": "VRApprox+decayApprox-w40",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_mfeBoundq0.1_": "VRApprox+decayApprox-UW5B0.1",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_mfeBoundq0.2_": "VRApprox+decayApprox-UW5B0.2",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_mfeBoundq0.3_": "VRApprox+decayApprox-UW5B0.3",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_mfeBoundq0.4_": "VRApprox+decayApprox-UW5B0.4",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_mfeBoundq_": "VRApprox+decayApprox-UW5B0.5",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_mfe2_": "VRApprox+decayApprox-UW5",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_mfeBoundq0.1_": "VRApprox+decayApprox-UW10B0.1",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_mfeBoundq0.2_": "VRApprox+decayApprox-UW10B0.2",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_mfeBoundq0.3_": "VRApprox+decayApprox-UW10B0.3",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_mfeBoundq0.4_": "VRApprox+decayApprox-UW10B0.4",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_mfeBoundq_": "VRApprox+decayApprox-UW10B0.5",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_mfe2_": "VRApprox+decayApprox-UW10",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW15AO_decay_mfeBoundq0.1_": "VRApprox+decayApprox-UW15B0.1",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW15AO_decay_mfeBoundq0.2_": "VRApprox+decayApprox-UW15B0.2",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW15AO_decay_mfeBoundq0.3_": "VRApprox+decayApprox-UW15B0.3",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW15AO_decay_mfeBoundq0.4_": "VRApprox+decayApprox-UW15B0.4",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW15AO_decay_mfeBoundq_": "VRApprox+decayApprox-UW15B0.5",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW15AO_decay_mfe2_": "VRApprox+decayApprox-UW15", # use main2, not mnist2
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW20AO_decay_mfeBoundq0.1_": "VRApprox+decayApprox-UW20B0.1",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW20AO_decay_mfeBoundq0.2_": "VRApprox+decayApprox-UW20B0.2",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW20AO_decay_mfeBoundq0.3_": "VRApprox+decayApprox-UW20B0.3",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW20AO_decay_mfeBoundq0.4_": "VRApprox+decayApprox-UW20B0.4",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW20AO_decay_mfeBoundq_": "VRApprox+decayApprox-UW20B0.5",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW20AO_decay_mfe2_": "VRApprox+decayApprox-UW20",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW30AO_decay_mfe2_": "VRApprox+decayApprox-UW30",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW40AO_decay_mfe2_": "VRApprox+decayApprox-UW40",
# #"3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_mfeBoundq0.05_": "VRApprox+decayApprox-UW5B0.05",
# #"3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_mfeBoundq0.05_": "VRApprox+decayApprox-UW10B0.05",# use main2, not mnist2
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW30AO_decay_mfeBoundq_": "VRApprox+decayApprox-UW30B0.5",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraUW40AO_decay_mfeBoundq_": "VRApprox+decayApprox-UW40B0.5",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw30_decay_m_": "VRApprox+decayApproxw30",
# #"1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextra_b0.9_m_": "VRApprox+b0.9",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraAO_b0.5_mfe2_": "VRApprox+b0.5",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraAO_b0.6_mfe2_": "VRApprox+b0.6",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraAO_b0.7_mfe2_": "VRApprox+b0.7",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraAO_b0.8_mfe2_": "VRApprox+b0.8",
# "3task_nnnnn_lessVennc0.3uv0.90.1_DiffstaleNoextraAO_b0.9_mfe2_": "VRApprox+b0.9",
# "3task_nnnnn_lessVennc0.3uv0.90.1_full_mfe2_": "full participation",
# "3task_nnnnn_lessVennc0.3uv0.90.1_random_mfe2_": "random",


# "1task_nnnnn_lessVennc0.1uv0.90.1_Diffstale_Ob_f_": "VR+optimal_b",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_decay_f_": "VRApprox+decayApprox",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw5AO_decay_f_": "VRApprox+decayApprox-w5",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw10AO_decay_f_": "VRApprox+decayApprox-w10",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw15AO_decay_f_": "VRApprox+decayApprox-w15", # use main2, not mnist2
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw20AO_decay_f_": "VRApprox+decayApprox-w20",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw30AO_decay_f_": "VRApprox+decayApprox-w30",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraw40AO_decay_f_": "VRApprox+decayApprox-w40",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW5AO_decay_f_": "VRApprox+decayApprox-UW5",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW10AO_decay_f_": "VRApprox+decayApprox-UW10",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW15AO_decay_f_": "VRApprox+decayApprox-UW15", # use main2, not mnist2
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW20AO_decay_f_": "VRApprox+decayApprox-UW20",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW30AO_decay_f_": "VRApprox+decayApprox-UW30",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraUW40AO_decay_f_": "VRApprox+decayApprox-UW40",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.5_f_": "VRApprox+b0.5",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.6_f_": "VRApprox+b0.6",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.7_f_": "VRApprox+b0.7",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.8_f_": "VRApprox+b0.8",
# "1task_nnnnn_lessVennc0.1uv0.90.1_DiffstaleNoextraAO_b0.9_f_": "VRApprox+b0.9",
# "1task_nnnnn_lessVennc0.1uv0.90.1_full_f_": "full participation",
# "1task_nnnnn_lessVennc0.1uv0.90.1_random_f_": "random",

# # # fashion-mnist, extreme non-iid: class ratio=0.1
# "1task_nnnnn_class0.1c0.1uv0.90.1_Diffstale_Ob_f_": "VR+optimal_b",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2real-5_f_": "VROb-UW2-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2m40-5_f_": "VROb-UW2m40-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2real-10_f_": "VROb-UW2-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2m65-10_f_": "VROb-UW2m65-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2real-20_f_": "VROb-UW2-20",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2real-40_f_": "VROb-UW2-40",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2-5_f_": "VROb-UW3-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW3m40-5_f_": "VROb-UW3m40-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2-10_f_": "VROb-UW3-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW3m65-10_f_": "VROb-UW3m65-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2-20_f_": "VROb-UW3-20",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_Ob_UW2-40_f_": "VROb-UW3-40",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_decay_f_": "VRApprox+decayApprox",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW5AO_decay_f_": "VRApprox-UW1-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW1m40-5AO_decay_f_": "VRApprox-UW1m40-5",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW10AO_decay_f_": "VRApprox-UW1-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW1m65-10AO_decay_f_": "VRApprox-UW1m65-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW15AO_decay_f_": "VRApprox-UW1-15",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW20AO_decay_f_": "VRApprox-UW1-20",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW30AO_decay_f_": "VRApprox-UW1-30",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW40AO_decay_f_": "VRApprox-UW1-40",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUWW5AO_decay_f_": "VRApprox-UW2-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW2m40-5AO_decay_f_": "VRApprox-UW2m40-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUWW10AO_decay_f_": "VRApprox-UW2-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW2m65-10AO_decay_f_": "VRApprox-UW2m65-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUWW15AO_decay_f_": "VRApprox-UW2-15",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUWW20AO_decay_f_": "VRApprox-UW2-20",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUWW30AO_decay_f_": "VRApprox-UW2-30",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUWW40AO_decay_f_": "VRApprox-UW2-40",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3-5AO_decay_f_": "VRApprox-UW3-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3m40-5AO_decay_f_": "VRApprox-UW3m40-5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3-10AO_decay_f_": "VRApprox-UW3-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3m65-10AO_decay_f_": "VRApprox-UW3m65-10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3-15AO_decay_f_": "VRApprox-UW3-15",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3-20AO_decay_f_": "VRApprox-UW3-20",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3-30AO_decay_f_": "VRApprox-UW3-30",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW3-40AO_decay_f_": "VRApprox-UW3-40",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K20_decay_f_": "VRApprox-K20",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K30_decay_f_": "VRApprox-K30",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K40_decay_f_": "VRApprox-K40",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K50_decay_f_": "VRApprox-K50",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K60_decay_f_": "VRApprox-K60",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K70_decay_f_": "VRApprox-K70",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K80_decay_f_": "VRApprox-K80",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K90_decay_f_": "VRApprox-K90",
# "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextra_K100_decay_f_": "VRApprox-K100",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw5AO_decay_f_": "VRApprox-w5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw10AO_decay_f_": "VRApprox-w10",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw15AO_decay_f_": "VRApprox-w15",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw20AO_decay_f_": "VRApprox-w20",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw30AO_decay_f_": "VRApprox-w30",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw40AO_decay_f_": "VRApprox-w40",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.5_f_": "VRApprox+b0.5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.6_f_": "VRApprox+b0.6",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.7_f_": "VRApprox+b0.7",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.8_f_": "VRApprox+b0.8",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.9_f_": "VRApprox+b0.9",
# "1task_nnnnn_class0.1c0.1uv0.90.1_uniform_b0.5_f_": "uniform+b0.5",
# "1task_nnnnn_class0.1c0.1uv0.90.1_uniform_b0.6_f_": "uniform+b0.6",
# "1task_nnnnn_class0.1c0.1uv0.90.1_uniform_b0.7_f_": "uniform+b0.7",
# "1task_nnnnn_class0.1c0.1uv0.90.1_uniform_b0.8_f_": "uniform+b0.8",
# "1task_nnnnn_class0.1c0.1uv0.90.1_uniform_b0.9_f_": "uniform+b0.9",
# "1task_nnnnn_class0.1c0.1uv0.90.1_uniform_Ob_f_": "uniform+Ob",
# "1task_nnnnn_class0.1c0.1uv0.90.1_full_f_": "full participation",
# "1task_nnnnn_class0.1c0.1uv0.90.1_random_f_": "random",
# "1task_nnnnn_class0.1c0.1uv0.90.1_GVR_f_": "GVR",
# "1task_nnnnn_class0.1c0.1uv0.90.1_GVRApprox_f_": "GVRApprox",

# # fashion-mnist, extreme non-iid: class ratio=0.1
# "5task_nnnnn_class0.1c0.3uv0.90.1_Diffstale_Ob_f_": "VR+optimal_b",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraAO_decay_f_": "VRApprox+decayApprox",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_f_": "VRApprox-UW5",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_f_": "VRApprox-UW10",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW20AO_decay_f_": "VRApprox-UW20",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW30AO_decay_f_": "VRApprox-UW30",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUWSD5AO_decay_f_": "VRApprox-UWSD5",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUWSD10AO_decay_f_": "VRApprox-UWSD10",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUWSD20AO_decay_f_": "VRApprox-UWSD20",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUWSD30AO_decay_f_": "VRApprox-UWSD30",
# #"1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraUW40AO_decay_f_": "VRApprox-UW40",
# # "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW5AO_decay_B0.3_": "VRApprox-UW5B0.3",
# # "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW10AO_decay_B0.3_": "VRApprox-UW10B0.3",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraw5AO_decay_f_": "VRApprox-w5",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraw10AO_decay_f_": "VRApprox-w10",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraw20AO_decay_f_": "VRApprox-w20",
# #"1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw20AO_decay_f_": "VRApprox-w20",
# "5task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraw30AO_decay_f_": "VRApprox-w30",
# #"1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraw40AO_decay_f_": "VRApprox-w40",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.5_f_": "VRApprox+b0.5",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.6_f_": "VRApprox+b0.6",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.7_f_": "VRApprox+b0.7",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.8_f_": "VRApprox+b0.8",
# # "1task_nnnnn_class0.1c0.1uv0.90.1_DiffstaleNoextraAO_b0.9_f_": "VRApprox+b0.9",
# "5task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.5_f_": "uniform+b0.5",
# "5task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.6_f_": "uniform+b0.6",
# "5task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.7_f_": "uniform+b0.7",
# "5task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.8_f_": "uniform+b0.8",
# "5task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.9_f_": "uniform+b0.9",
# "5task_nnnnn_class0.1c0.3uv0.90.1_full_f_": "full participation",
# "5task_nnnnn_class0.1c0.3uv0.90.1_random_f_": "random",
# "5task_nnnnn_class0.1c0.3uv0.90.1_FedVARP_f_": "FedVARP",
# # "5task_nnnnn_class0.1c0.3uv0.90.1_MILA_f_": "MIFA",
# # "5task_nnnnn_class0.1c0.3uv0.90.1_SCAFFOLD_f_": "SCAFFOLD",
# #"1task_nnnnn_class0.1c0.1uv0.90.1_GVR_f_": "GVR",
# #"1task_nnnnn_class0.1c0.1uv0.90.1_GVRApprox_f_": "GVRApprox",

# # 3 task fashion-mnist # accuracy too low
# # fashion-mnist, extreme non-iid: class ratio=0.1
# "3task_nnnnn_class0.1c0.3uv0.90.1_Diffstale_Ob_f_": "VR+optimal_b",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraAO_decay_f_": "VRApprox+decayApprox",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW1-5AO_decay_f_": "VRApprox-UW1-5",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW1-10AO_decay_f_": "VRApprox-UW1-10",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW1-20AO_decay_f_": "VRApprox-UW1-20",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW1-40AO_decay_f_": "VRApprox-UW1-40",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW2-5AO_decay_f_": "VRApprox-UW2-5",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW2-10AO_decay_f_": "VRApprox-UW2-10",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW2-20AO_decay_f_": "VRApprox-UW2-20",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW2-40AO_decay_f_": "VRApprox-UW2-40",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW3-5AO_decay_f_": "VRApprox-UW3-5",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW3-10AO_decay_f_": "VRApprox-UW3-10",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW3-20AO_decay_f_": "VRApprox-UW3-20",
# "3task_nnnnn_class0.1c0.3uv0.90.1_DiffstaleNoextraUW3-40AO_decay_f_": "VRApprox-UW3-40",
# "3task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.7_f_": "uniform+b0.7",
# "3task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.8_f_": "uniform+b0.8",
# "3task_nnnnn_class0.1c0.3uv0.90.1_uniform_b0.9_f_": "uniform+b0.9",
# "3task_nnnnn_class0.1c0.3uv0.90.1_full_f_": "full participation",
# "3task_nnnnn_class0.1c0.3uv0.90.1_random_f_": "random",


# "1task_nnnnn_class0.3c0.1uv0.90.1_Diffstale_Ob_m_": "VR+optimal_b",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraAO_decay_m_": "VRApprox+decayApprox",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW1-5AO_decay_m_": "VRApprox-UW1-5",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW1m40-5AO_decay_m_": "VRApprox-UW1m40-5",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW1-10AO_decay_m_": "VRApprox-UW1-10",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW1m65-10AO_decay_m_": "VRApprox-UW1m65-10",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW1-20AO_decay_m_": "VRApprox-UW1-20",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW1-40AO_decay_m_": "VRApprox-UW1-40",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW2-5AO_decay_m_": "VRApprox-UW2-5",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW2m40-5AO_decay_m_": "VRApprox-UW2m40-5",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW2-10AO_decay_m_": "VRApprox-UW2-10",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW2m65-10AO_decay_m_": "VRApprox-UW2m65-10",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW2-20AO_decay_m_": "VRApprox-UW2-20",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW2-40AO_decay_m_": "VRApprox-UW2-40",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW3-5AO_decay_m_": "VRApprox-UW3-5",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW3m40-5AO_decay_m_": "VRApprox-UW3m40-5",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW3-10AO_decay_m_": "VRApprox-UW3-10",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW3m65-10AO_decay_m_": "VRApprox-UW3m65-10",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW3-20AO_decay_m_": "VRApprox-UW3-20",
# "1task_nnnnn_class0.3c0.1uv0.90.1_DiffstaleNoextraUW3-40AO_decay_m_": "VRApprox-UW3-40",
# "1task_nnnnn_class0.3c0.1uv0.90.1_full_m_": "full participation",
# "1task_nnnnn_class0.3c0.1uv0.90.1_random_m_": "random",
# "1task_nnnnn_class0.3c0.1uv0.90.1_uniform_b0.5_m_": "uniform+b0.5",
# "1task_nnnnn_class0.3c0.1uv0.90.1_uniform_b0.6_m_": "uniform+b0.6",
# "1task_nnnnn_class0.3c0.1uv0.90.1_uniform_b0.7_m_": "uniform+b0.7",
# "1task_nnnnn_class0.3c0.1uv0.90.1_uniform_b0.8_m_": "uniform+b0.8",
# "1task_nnnnn_class0.3c0.1uv0.90.1_uniform_b0.9_m_": "uniform+b0.9",
# "1task_nnnnn_class0.3c0.1uv0.90.1_uniform_Ob_m_": "uniform+Ob",

# "1task_nnnnn_class0.2c0.1uv0.90.1_Diffstale_Ob_e_": "VR+optimal_b",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraAO_decay_e_": "VRApprox+decayApprox",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW1-5AO_decay_e_": "VRApprox-UW1-5",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW1m40-5AO_decay_e_": "VRApprox-UW1m40-5",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW1-10AO_decay_e_": "VRApprox-UW1-10",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW1m65-10AO_decay_e_": "VRApprox-UW1m65-10",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW1-20AO_decay_e_": "VRApprox-UW1-20",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW1-40AO_decay_e_": "VRApprox-UW1-40",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW2-5AO_decay_e_": "VRApprox-UW2-5",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW2m40-5AO_decay_e_": "VRApprox-UW2m40-5",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW2-10AO_decay_e_": "VRApprox-UW2-10",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW2m65-10AO_decay_e_": "VRApprox-UW2m65-10",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW2-20AO_decay_e_": "VRApprox-UW2-20",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW2-40AO_decay_e_": "VRApprox-UW2-40",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW3-5AO_decay_e_": "VRApprox-UW3-5",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW3m40-5AO_decay_e_": "VRApprox-UW3m40-5",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW3-10AO_decay_e_": "VRApprox-UW3-10",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW3m65-10AO_decay_e_": "VRApprox-UW3m65-10",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW3-20AO_decay_e_": "VRApprox-UW3-20",
# "1task_nnnnn_class0.2c0.1uv0.90.1_DiffstaleNoextraUW3-40AO_decay_e_": "VRApprox-UW3-40",
# "1task_nnnnn_class0.2c0.1uv0.90.1_full_e_": "full participation",
# "1task_nnnnn_class0.2c0.1uv0.90.1_random_e_": "random",
# "1task_nnnnn_class0.2c0.1uv0.90.1_uniform_b0.5_e_": "uniform+b0.5",
# "1task_nnnnn_class0.2c0.1uv0.90.1_uniform_b0.6_e_": "uniform+b0.6",
# "1task_nnnnn_class0.2c0.1uv0.90.1_uniform_b0.7_e_": "uniform+b0.7",
# "1task_nnnnn_class0.2c0.1uv0.90.1_uniform_b0.8_e_": "uniform+b0.8",
# "1task_nnnnn_class0.2c0.1uv0.90.1_uniform_b0.9_e_": "uniform+b0.9",
# "1task_nnnnn_class0.2c0.1uv0.90.1_uniform_Ob_e_": "uniform+Ob",

# fixed sampling
# "1task_nnnnn_class0.2c0.1uvNo_a3.0_b0.5_e_": "+b0.5",
# "1task_nnnnn_class0.2c0.1uvNo_a3.0_b0.6_e_": "uniform+b0.6",
# "1task_nnnnn_class0.2c0.1uvNo_a3.0_b0.7_e_": "uniform+b0.7",
# "1task_nnnnn_class0.2c0.1uvNo_a3.0_b0.8_e_": "uniform+b0.8",
# "1task_nnnnn_class0.2c0.1uvNo_a3.0_b0.9_e_": "uniform+b0.9",
# "1task_nnnnn_class0.2c0.1uvNo_a3.0_Ob_e_": "uniform+Ob",

"1task_nnnnn_class0.3c0.1uvNo_a5.0_Ob_icdcs_": "StaleVR",
"1task_nnnnn_class0.3c0.1uvNo_a5.0_b0.5_icdcs_": "FedStale(beta=0.6)",
"1task_nnnnn_class0.3c0.1uvNo_a5.0_b0.6_icdcs_": "FedStale(beta=0.7)",
"1task_nnnnn_class0.3c0.1uvNo_a5.0_b0.7_icdcs_": "FedStale(beta=0.8)",
"1task_nnnnn_class0.3c0.1uvNo_a5.0_b0.8_icdcs_": "FedStale(beta=0.9)",
"1task_nnnnn_class0.3c0.1uvNo_a5.0_b0.9_icdcs_": "FedVARP",
# available seeds:
}
all_rounds=150
# 12 13
seed_list = [12,13,14,15,16,17,18]
horizontal = [1, 0, 0, 1, 0, 0]
tasknum= 3
curve_compare = []
# 17 16 15
line_list = ['-', '-', '-', '-', '-','-','-','--','-','-','-','-', '--','-','-','-','-','--','-','-','-','-','-','-','-','--','-','-','-','-','-','-','-','-','--','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
marker_list =['o', 's', 'v','^', 'p','D','X', 'P','^', 'p','D','X', 'P','p','D','X', 'P','p','D','X', 'P','p','D','X', 'P','D','X', 'P','p','D','X', 'P','X', 'P','P','X','P','p','D','X', 'P','X', 'P','P','X','D','X', 'P','X', 'P','P','X']
# sd 21 is good,
# sd 19, 20 is bad
finalPath = f'./result/1task_nnn_test_13'

# make figure wide=8, height=5

fig_avg = plt.figure(figsize=(8,5))
fig_min = plt.figure()
ax_avg = fig_avg.add_subplot(1, 1, 1)
ax_min = fig_min.add_subplot(1, 1, 1)
all_algorithm_curve = []
keys_list = []
all_count_list = []
cnt = 0
for key in extra_folder:
    current_folder = {}
    current_folder[key] = extra_folder[key]
    keys_list.append(extra_folder[key])
    global_avg_acc = 0
    global_min_acc = 0
    global_max_acc = 0
    best10_avg = 0
    worst10_avg = 0
    var_avg = 0
    client_var_avg = 0
    allocation_var = 0
    curve = []
    # if ms, use seed_list_ms
    # if 'ms' in key:
    #     seed_list = seed_list_ms.copy()
    # elif 'alpha' in key:
    #     seed_list = seed_list_alpha.copy()
    # elif 'AS' in key:
    #     seed_list = seed_list_AS.copy()
    algor_seed = seed_list.copy()
    count_list = []

    for seed in seed_list:
        #seed = list(current_folder.keys())[0][-2:]
        # plot allocation
        # if tasknum > 1:
        #     allo_var, client_count_all = allocation(current_folder, seed)
        #     if type(client_count_all) is not int:
        #         count_list.append(client_count_all)
        #         allocation_var += allo_var
        # else:
        #     allocation_var = 0


        exp_array = load_extra_folder(current_folder, seed, header='localAcc_')
        # reshape 1 2 40 to 2 40
        if exp_array.shape[-1] == 0:
            algor_seed.remove(seed)
            continue
        if exp_array.shape[-1] == all_rounds:
            exp_array = exp_array[:, :, :, -1].reshape(exp_array.shape[1], exp_array.shape[2])
        else:
            exp_array = exp_array.reshape(exp_array.shape[1], exp_array.shape[2])
        path_plot = os.path.join('./result', key)
        algo_name = current_folder.values()
        tasknum = exp_array.shape[0]
        clients_num = exp_array.shape[1]
        # compute average
        exp_array_avg = np.mean(exp_array, axis=0)


        client_var = np.var(exp_array, axis=1)
        client_var_avg += np.mean(client_var)
        # compute entropy
        # normalize to prob using softmax
        # softmax
        exp_array2 = np.exp(exp_array)
        exp_array2 = exp_array2 / np.sum(exp_array2)
        exp_array_entropy = -np.sum(exp_array2 * np.log(exp_array2))
        # compute KL divergence
        P_a = exp_array2.reshape(-1)

        P_uniform = np.ones_like(P_a) / len(P_a)
        kl = np.sum(P_a * np.log(P_a / P_uniform))



        # compute minimum
        # sort avg and min
        exp_array_avg = np.sort(exp_array_avg)
        worst10_avg += np.mean(exp_array_avg[:int(clients_num * 0.2)])
        worst10_std = np.std(exp_array_avg[:int(clients_num * 0.1)])
        best10_avg += np.mean(exp_array_avg[-int(clients_num * 0.2):])
        best10_std = np.std(exp_array_avg[-int(clients_num * 0.1):])

        # get global final acc
        exp_array = load_extra_folder(current_folder, seed, header='mcf_i_globalAcc_')
        exp_array = exp_array.reshape(exp_array.shape[1], exp_array.shape[2])
        curve.append(exp_array)

        last = -1
        global_avg_acc += np.mean(exp_array[:, last-20:])
        global_min_acc += np.mean(np.min(exp_array[:, last-20:], axis=1))
        global_max_acc += np.mean(np.max(exp_array[:, last-20:], axis=1))
        var_avg += np.mean(np.var(exp_array))
    global_avg_acc /= len(algor_seed)
    global_min_acc /= len(algor_seed)
    global_max_acc /= len(algor_seed)
    best10_avg /= len(algor_seed)
    worst10_avg /= len(algor_seed)
    var_avg /= len(algor_seed)
    client_var_avg /= len(algor_seed)
    allocation_var /= len(algor_seed)
    algoName = next(iter(algo_name))
    #print(f"{algoName: <25}: \t Global avg acc: {global_avg_acc:.3f}, max: {global_max_acc:.3f}, min: {global_min_acc:.3f}, var_avg: {np.sqrt(var_avg):.3f}")
    print(
        f"{algoName: <25}: \t Global avg acc: {global_avg_acc:.3f}, max: {global_max_acc:.3f}, min: {global_min_acc:.3f}, var_avg: {np.sqrt(var_avg):.3f}")
    #print(f"{algoName: <10}: \t worst20% {worst10_avg:.3f}, best20% {best10_avg:.3f}; Global acc: {global_avg_acc:.3f} entropy: {exp_array_entropy:.3f}, KL{kl:.4f} client_var: {client_var_avg:.3f}")
    #averge the curve
    curve = np.array(curve) # shape: seed tasknum numRounds
    # for 5 tasks experiment only
    # curve = curve[:,:4,:]
    # tasknum = 4
    x = np.arange(curve.shape[-1])
    # get the upper and lower bound of the curve
    aver_each_seed_curve = np.mean(curve, axis=1)

    all_algorithm_curve.append(np.mean(curve, axis=0))
    all_count_list.append(count_list)
    curve_avg = np.mean(curve, axis=(0,1)).reshape(-1)
    curve_min = np.min(np.mean(curve, axis=0).reshape(tasknum, -1), axis=0).reshape(-1)

    curve_upper = curve_avg + aver_each_seed_curve.std(axis=0)
    curve_lower = curve_avg - aver_each_seed_curve.std(axis=0)

    ax_avg.plot(x, curve_avg, label=next(iter(algo_name)), linestyle=line_list[cnt], marker=marker_list[cnt], markevery=20, linewidth=1.5, markersize=8)
    final_acc = curve_avg[-1]
    if horizontal[cnt] == 1:
        curve_compare.append(curve_avg)
        ax_avg.axhline(y=final_acc, color='gray', linestyle='--', linewidth=0.8)
        ax_avg.text(-21, final_acc, f"{final_acc:.2f}", va='center', ha='left', fontsize=15)

    cnt+=1
    # ax_avg.fill_between(x, curve_lower, curve_upper, alpha=0.2)


    ax_min.plot(x, curve_min, label=next(iter(algo_name)), linestyle=line_list[cnt], marker=marker_list[cnt], markevery=20, linewidth=1.5, markersize=8)

# plot each task
# if tasknum > 1:
#     plot_each_task(all_algorithm_curve, keys_list)
#     plot_allocation_count(all_count_list, keys_list)

# ax_avg.legend(fontsize=12, frameon=False)
# # make label size larger
# ax_avg.set_xlabel('Num. Global Iterations', fontsize=20)
# ax_avg.set_ylabel('Accuracy', fontsize=20)
# ax_avg.set_title(f'Avg Accuracy over {tasknum} Models', fontsize=20)
# ax_avg.tick_params(axis='both', which='major', labelsize=20)
# # make the figure is tight
#
# ax_min.legend(fontsize=12, frameon=False)
# ax_min.set_xlabel('Num. Global Iterations', fontsize=20)
# ax_min.set_ylabel('Accuracy', fontsize=20)
# ax_min.set_title(f'Min Accuracy over {tasknum} Models', fontsize=20)
# ax_min.tick_params(axis='both', which='major', labelsize=20)
# limit y between 0.2 to 0.8
ax_avg.set_ylim([0.2, 0.8])
ax_avg.legend(frameon=False, fontsize=14) # ncol=2
# make label size larger
ax_avg.set_xlabel('Num. Global Iterations', fontsize=20)
ax_avg.set_ylabel('Accuracy', fontsize=20)
#ax_avg.set_title(f'Avg Accuracy over {tasknum} Models', fontsize=20)
ax_avg.set_title(f'Accuracy', fontsize=20)
ax_avg.tick_params(axis='both', which='major', labelsize=20)
# make the figure is tight

ax_min.legend(frameon=False)
ax_min.set_xlabel('Num. Global Iterations')
ax_min.set_ylabel('Accuracy', fontsize=20)
ax_min.set_title(f'Min Accuracy over {tasknum} Models')
#ax_min.tick_params(axis='both', which='major', labelsize=20)


fig = ax_avg.get_figure()
# save to 3task_nnn_u{u_value}d{d_value}_random_11/global_avg_acc.png
fig.tight_layout()
fig.savefig(finalPath+'/global_avg_acc.pdf', format="pdf", bbox_inches="tight")
fig = ax_min.get_figure()
fig.tight_layout()
fig.savefig(finalPath+'/global_min_acc.pdf', format="pdf", bbox_inches="tight")


# accuracy_levels from 0.2 to 0.8, linespace
# accuracy_levels = np.linspace(0.2, 0.55, 1000)
# gap = []
# for level in accuracy_levels:
# # find the first index that is greater than level
#     GVR_round = 0
#     SMVR_round = 0
#     for i in range(len(curve_compare[0])): # GVR
#         if curve_compare[0][i] > level:
#             GVR_round = i
#             break
#     for i in range(len(curve_compare[0])): # SMVR
#         if curve_compare[1][i] > level:
#             SMVR_round = i
#             break
#     gap.append(SMVR_round - GVR_round)
# print(gap)
# # plot gap
# fig_gap = plt.figure()
# ax_gap = fig_gap.add_subplot(1, 1, 1)
# ax_gap.plot(np.arange(150), curve_compare[0]-curve_compare[1])
# ax_gap.set_xlabel('Accuracy Level')
# ax_gap.set_ylabel('Round Gap')
# fig = ax_gap.get_figure()
# fig.tight_layout()
# #fig.savefig(finalPath+'/round_gap.pdf', format="pdf", bbox_inches="tight")
# plt.show()