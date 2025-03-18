import torch
import numpy as np
import random
import pickle
from utility.config import optimizer_config
import copy


def get_gradient_norm(weights_this_round, weights_next_round, lr):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_this_round[name] - weights_next_round[name]).cpu() for name in weights_this_round}
    # Calculate the L2 norm of the weight differences
    # bound in case appear nan
    norm = sum(torch.norm(diff, p=2) ** 2 for diff in weight_diff.values()) ** 0.5 / lr
    norm.item()
    if torch.isnan(norm):
        norm = torch.tensor(0.0)
    return norm.item(), weight_diff


def weight_minus(weights_A, weights_B):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_A[name] - weights_B[name]).cpu() for name in weights_A}
    # Calculate the L2 norm of the weight differences
    return weight_diff


def weight_add(weights_A, weights_B):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_A[name] + weights_B[name]).cpu() for name in weights_A}
    # Calculate the L2 norm of the weight differences
    return weight_diff


def weight_product(weights_A, weights_B, epsilon=1e-7):
    # Calculate weight_A * weight_B (inner product, return a scalar)
    weight_prod = sum(torch.sum(weights_A[name] * weights_B[name]) for name in weights_A)

    # Replace NaN with a small value and bound very small values
    if torch.isnan(weight_prod) or torch.abs(weight_prod) < epsilon:
        weight_prod = epsilon  # Set to a small non-zero value

    return weight_prod


def weight_product_lastLayer(weights_A, weights_B, epsilon=1e-7):
    # Calculate weight_A * weight_B only for their last layer parameters
    last_layer_name = list(weights_A.keys())[-1]
    weight_prod = torch.sum(weights_A[last_layer_name] * weights_B[last_layer_name])

    # Replace NaN with a small value and bound very small values
    if torch.isnan(weight_prod) or torch.abs(weight_prod) < epsilon:
        weight_prod = epsilon  # Set to a small non-zero value

    return weight_prod

def weight_norm(weights_A):
    # get gradient by subtracting weights_next_round from weights_this_round
    # Calculate weight_A * weight_B (inner product, return a scalar)
    weight_norm = sum(torch.sum(weights_A[name]) for name in weights_A)
    return weight_norm



def zero_shapelike(weights):
    # get gradient by subtracting weights_next_round from weights_this_round
    zero_weights = {name: torch.zeros_like(tensor).cpu() for name, tensor in weights.items()}
    return zero_weights


def newupdate(weights, b0):
    weight_new = {name: (weights[name]*b0).cpu() for name in weights}
    return weight_new


def stale_decay(weights, b):
    weight_decay = {name: (weights[name]*b).cpu() for name in weights}
    return weight_decay


def append_to_pickle(file_path, new_data):
    # Step 1: Load existing data from the pickle file
    try:
        with open(file_path, 'rb') as file:
            existing_data = pickle.load(file)
    except FileNotFoundError:
        existing_data = []
    except EOFError:
        existing_data = []

    # Ensure the loaded data is a list (or another suitable data structure)
    if not isinstance(existing_data, list):
        raise ValueError("Existing data is not a list")

    # Step 2: Append new data
    existing_data.append(new_data)

    # Step 3: Save the updated data back to the pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(existing_data, file)


def power_gradient_norm(gradient_norm, tasks_local_training_loss, args, dis):
    gradient_norm_power = gradient_norm
    return gradient_norm_power


def get_optimal_sampling_single(m, gradient_record): # gradient record is norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    sample_num = m  # m in the paper
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record)

    all_gradients = gradient_record.copy()
    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(all_gradients)

    n = all_clients_num
    m = sample_num

    l = n - m + 1
    best_l = l
    if m == 0: # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(all_gradients[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / all_gradients[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_i = np.zeros(all_clients_num)
    sum_upto_l = sum(all_gradients[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            p_i[sorted_indices[i]] = 1
        else:
            p_i[sorted_indices[i]] = (m + best_l - n) * all_gradients[sorted_indices[i]] / sum_upto_l

    return p_i


def get_optimal_sampling_tasks(m, gradient_record): # gradient record is norm
    sample_num = int(m)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])

    all_gradients = gradient_record.copy()

    client_gradients_sumTasks = np.zeros(all_clients_num) # this is M_i in the proof
    for client_index in range(all_clients_num):
        for task_index in range(tasks_num):
            client_gradients_sumTasks[client_index] += all_gradients[task_index][client_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(client_gradients_sumTasks)
    print(sorted_indices)

    n = all_clients_num
    m = sample_num

    l = n - m + 1
    print(l)
    best_l = l
    if m == 0: # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / client_gradients_sumTasks[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_s_i = np.zeros((tasks_num, all_clients_num))
    sum_upto_l = sum(client_gradients_sumTasks[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = all_gradients[task_index][sorted_indices[i]] / client_gradients_sumTasks[sorted_indices[i]]
        else:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = (m + best_l - n) * all_gradients[task_index][sorted_indices[i]] / sum_upto_l
    return p_s_i


def B_neq_Prob(gradients_i, ki, task_num):
    p_i = get_optimal_sampling_single(m=ki, gradient_record=gradients_i)
    return p_i

def update_matrices(client_num, task_num, ki, active_num, all_gradients, set_matrix, p_si_matrix, ineq_matrix, evaluation_matrix):
    for i in range(client_num):
        for s in range(task_num):
            if set_matrix[s, i] == 1:
                # if set_matrix[:, i] all are zero, means it is in B_neq set
                B_neq_sum = 0
                Psi1_sum = 0
                for j in range(client_num):
                    if sum(set_matrix[:, j]) == 0:
                        B_neq_sum += ki[j]
                    # else if set_matrix[:, i] exist zero, but not all zero
                    elif np.all(set_matrix[:, j]) == 0:
                        # count how many zero in set_matrix[:, j]
                        Psi1_sum += task_num - np.sum(set_matrix[:, j])
                p_si_matrix[s, i] = (active_num - B_neq_sum - Psi1_sum) * all_gradients[s, i] / np.sum(all_gradients * set_matrix)
    # update ineq_matrix and evaluation_matrix
    ineq_matrix[0] = np.sum(p_si_matrix, axis=0)
    ineq_matrix[1:] = p_si_matrix
    evaluation_matrix[0] = ineq_matrix[0] <= (ki+1e-7)
    evaluation_matrix[1:] = evaluation_matrix[1:] = ineq_matrix[1:] <= 1
    return p_si_matrix, ineq_matrix, evaluation_matrix


def optimal_sampling2(client_num, task_num, all_gradients, active_num, ki):
    ineq_matrix = np.zeros((task_num+1, client_num))  # use the first one to store \sum<ki
    evaluation_matrix = np.ones((task_num+1, client_num))
    p_si_matrix = np.zeros((task_num, client_num))
    set_matrix = np.ones((task_num, client_num))
    # intialize: assume all p_si not breaks the rule
    # p_si_matrix[s][i] = active_num * all_gradients[s][i] / np.sum(all_gradients)
    p_si_matrix = active_num * np.array(all_gradients) / np.sum(all_gradients)
    # update ineq_matrix
    ineq_matrix[0] = np.sum(p_si_matrix, axis=0)
    ineq_matrix[1:] = p_si_matrix
    # update set_matrix
    # if ineq_matrix[0][i] > ki, set_matrix[0][i] = 0
    # for s=[1:], if ineq_matrix[s][i] > 1, set_matrix[s][i] = 0
    evaluation_matrix[0] = ineq_matrix[0] <= ki
    evaluation_matrix[1:] = ineq_matrix[1:] <= 1
    # easiest case: set_matrix[0,i]==0, and np.all(set_matrix[1:,i])==1
    easiest_index = []
    easiest_done = False
    while easiest_done is False:
        for i in range(client_num):
            temp = evaluation_matrix[1:,i]
            if evaluation_matrix[0][i] == 0 and np.all(temp) == 1:
                p_si_matrix[:, i] = B_neq_Prob(all_gradients[:, i], ki[i], task_num)
                set_matrix[:, i] = 0
                easiest_index.append(i)
                p_si_matrix, ineq_matrix, evaluation_matrix  = update_matrices(client_num, task_num, ki, active_num, all_gradients,
                                         set_matrix, p_si_matrix, ineq_matrix, evaluation_matrix)
        # if all easiest cases are solved, then break
        cnt = 0
        for i in range(client_num):
            temp = evaluation_matrix[1:,i]
            if evaluation_matrix[0][i] == 0 and np.all(temp) == 1:
                # not finish
                easiest_done = False
                break
            else:
                cnt += 1
        if cnt == client_num:
            easiest_done = True
    # separate p_si to esaiest and not
    # record not-easiest index
    not_easiest_index = []
    for i in range(client_num):
        if i not in easiest_index:
            not_easiest_index.append(i)
    p_si_matrix_easiest = p_si_matrix[:, easiest_index]
    all_gradients_temp = np.delete(all_gradients, easiest_index, axis=1)
    sum_easiest = np.sum(p_si_matrix_easiest)
    active_num_temp = active_num - sum_easiest
    while np.all(evaluation_matrix) != 1:
        p_s_i = get_optimal_sampling_tasks(m=active_num_temp, gradient_record=all_gradients_temp)
        print("finish step 2.1, if not consider sum<ki, then we are good")
        # recover p_si_matrix
        p_si_matrix = np.zeros((task_num, client_num))
        p_si_matrix[:, easiest_index] = p_si_matrix_easiest
        p_si_matrix[:, not_easiest_index] = p_s_i
        # update evaluation
        ineq_matrix[0] = np.sum(p_si_matrix, axis=0)
        ineq_matrix[1:] = p_si_matrix
        evaluation_matrix[0] = ineq_matrix[0] <= ki+1e-7
        evaluation_matrix[1:] = ineq_matrix[1:] <= 1


        temp_evaluation2 = evaluation_matrix[0]
        if np.all(temp_evaluation2) != 1:
            gap = ineq_matrix[0] - ki
            # find the index of the largest gap
            i = np.argmax(gap)
            p_si_matrix[:, i] = B_neq_Prob(all_gradients[:, i], ki[i], task_num)
            set_matrix[:, i] = 0
            easiest_index.append(i)
            not_easiest_index.remove(i)
            p_si_matrix_easiest = p_si_matrix[:, easiest_index]
            all_gradients_temp = np.delete(all_gradients, easiest_index, axis=1)
            sum_easiest = np.sum(p_si_matrix_easiest)
            active_num_temp = active_num - sum_easiest
            print('finish step 2.2, consider sum<ki')


    # print('final evaluation_matrix\n', evaluation_matrix)
    return p_si_matrix

def get_optimal_sampling(chosen_processes, dis, gradient_record, args, client_task_ability, clients_process, venn_matrix, save_path): # gradient record is norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    sample_num = len(chosen_processes)  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])
    processes_num = sum(client_task_ability) # client_task_ability [4,2,1,..] client 1 has 4 processes, client 2 has 2 processes

    all_gradients = np.zeros((tasks_num, processes_num))

    if args.suboptimal: # set client_task_ability to all 1
        client_task_ability = np.ones(all_clients_num)


    for task_index in range(tasks_num):
        for process_index in range(processes_num):
        # from U to U~ in the paper
            client_index = clients_process[process_index]
            all_gradients[task_index][process_index] = gradient_record[task_index][client_index] * dis[task_index][client_index] / client_task_ability[client_index] * venn_matrix[task_index][client_index]
    process_gradients_sumTasks = np.zeros(processes_num) # this is M_i in the proof
    for process_index in range(processes_num):
        for task_index in range(tasks_num):
            process_gradients_sumTasks[process_index] += all_gradients[task_index][process_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(process_gradients_sumTasks)

    n = processes_num
    m = sample_num

    l = n - m + 1
    best_l = l
    if m == 0: # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(process_gradients_sumTasks[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / process_gradients_sumTasks[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_s_i = np.zeros((tasks_num, processes_num))
    sum_upto_l = sum(process_gradients_sumTasks[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = all_gradients[task_index][sorted_indices[i]] / process_gradients_sumTasks[sorted_indices[i]]
        else:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = (m + best_l - n) * all_gradients[task_index][sorted_indices[i]] / sum_upto_l

    allocation_result = np.zeros(processes_num, dtype=int)
    for process_idx in range(processes_num):
        if abs(1-np.sum(p_s_i[:, process_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, process_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num+1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, process_idx]
        allocation_result[process_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client) # appear NaN
    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_process_order = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    chosen_clients = [clients_process[i] for i in chosen_process_order]
    # get p_dict
    p_dict = []
    active_rate = len(chosen_clients)/all_clients_num
    if args.equalP or args.equalP2:
        for task_index in range(tasks_num):
            p_dict.append([active_rate for i in range(all_clients_num) if allocation_result[i] == task_index])
    else:
        for task_index in range(tasks_num):
            p_dict.append([p_s_i[task_index][i] * client_task_ability[clients_process[i]] for i in range(processes_num) if allocation_result[i] == task_index])

    # store p_s_i, process_gradients_sumTasks
    if args.optimal_sampling is True:
        type = 'OS'
    elif args.approx_optimal is True:
        type = 'AS'
    else:
        type = 'none'
    file_path = save_path + 'psi_'+type+'.pkl'
    append_to_pickle(file_path, p_s_i)
    file_path = save_path + 'gradient_'+type+'.pkl'
    append_to_pickle(file_path, process_gradients_sumTasks)
    file_path = save_path + 'k_' + type + '.pkl'
    append_to_pickle(file_path, l)

    # record gradient_record
    file_path = save_path + 'gradient_record_' + type + '.pkl'
    append_to_pickle(file_path, gradient_record)

    return clients_task, p_dict, chosen_clients

def get_clients_num_per_task(clients_task, tasks_num):
    clients_num_per_task = [0] * tasks_num # list with length of tasks_num
    for task_index in range(tasks_num):
        # count the number of clients for each task
        clients_num_per_task[task_index] = clients_task.tolist().count(task_index)
    return clients_num_per_task


import cvxpy as cp

def optimal_solver(client_num, task_num, all_gradients, ms_list):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    ms = np.array(ms_list)  # List of ms values
    # if ms exist 0, then the problem is infeasible
    # if 0 in ms, adjust ms to make it feasible
    if 0 in ms:
        for i in range(len(ms)):
            if ms[i] == 0:
                ms[i] = 1
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= 1]
    # Sum of p for each task across clients <= ms
    constraints += [cp.sum(p, axis=1) <= ms]
    # p >= 0 (implicitly satisfies p <= 1 due to the constraints above)
    constraints += [p >= 1e-10]
    constraints += [p <= 1]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    try:
        problem.solve(max_iters=1600000)
        p_optimal = p.value
    except:
        p_optimal = np.ones((S, N)) / S

    return p_optimal


def optimal_solver_delta(client_num, task_num, all_gradients, m, delta):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= 1]
    # Sum of p for all clients all task, sum(p)<m
    constraints += [cp.sum(p) <= m]
    # p >= 0 (implicitly satisfies p <= 1 due to the constraints above)
    constraints += [p >= delta]
    constraints += [p <= 1]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    try:
        problem.solve(max_iters=1600000)
        p_optimal = p.value
    except:
        print('optimize failed')
        p_optimal = np.ones((S, N)) / S

    return p_optimal

def tradeoff_solver(client_num, task_num, all_gradients, active_num, dis):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    m = active_num
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)
    unbalance_level = 1 / (N*np.min(dis))

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    punishment = dis
    U_punishment = cp.square(punishment)
    # objective:
    # min unbalance_level * U2/p + U_punishment/p
    objective = cp.Minimize(unbalance_level * cp.sum(cp.multiply(U2, cp.power(p, -1))) + cp.sum(cp.multiply(U_punishment, cp.power(p, -1))))
    # objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= 1]
    # Sum of all p for all task across clients <= m
    constraints += [cp.sum(p) <= m]
    # p >= 0 (implicitly satisfies p <= 1 due to the constraints above)
    constraints += [p >= 1e-10]
    constraints += [p <= 1]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    try:
        problem.solve(max_iters=1600000)
        p_optimal = p.value
    except:
        p_optimal = np.ones((S, N)) / S
        print('solver failed')
    return p_optimal


def communication_solver(client_num, task_num, all_gradients, active_num, ki):
    N = client_num  # Number of clients
    S = task_num    # Number of tasks
    U = np.array(all_gradients).reshape(task_num, client_num)  # Gradient record reshaped
    m = active_num
    # Define the variable to solve for
    p = cp.Variable((S, N), nonneg=True)
    # print("curvature of p", p.curvature)

    # Objective function
    # print("curvature of U", U.curvature)
    U2 = cp.square(U)
    # objective:
    # min unbalance_level * U2/p + U_punishment/p
    objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))
    # objective = cp.Minimize(cp.sum(cp.multiply(U2, cp.power(p, -1))))

    # Constraints
    constraints = []
    # Sum of p for each client across tasks <= 1
    constraints += [cp.sum(p, axis=0) <= ki]
    # Sum of all p for all task across clients <= m
    constraints += [cp.sum(p) <= m]
    # p >= 0 (implicitly satisfies p <= 1 due to the constraints above)
    constraints += [p >= 1e-10]
    constraints += [p <= 1]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    try:
        problem.solve(max_iters=1600000)
        p_optimal = p.value
    except:
        p_optimal = np.ones((S, N)) / (S + 1)

    return p_optimal


def get_optimal_sampling_cvx(clients_process, tasks_count, dis, gradient_record, client_task_ability, args, venn_matrix, save_path):
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    sample_num = np.sum(tasks_count)
    tasks_num = len(gradient_record)
    processes_num = len(gradient_record[0])

    ms_list = tasks_count
    all_gradients = gradient_record.copy()

    for task_index in range(tasks_num):
        for process_index in range(processes_num):
        # from U to U~ in the paper
            client_index = clients_process[process_index]
            all_gradients[task_index][process_index] = gradient_record[task_index][client_index] * dis[task_index][client_index] / client_task_ability[client_index] * venn_matrix[task_index][client_index]


    p_optimal = optimal_solver(client_num=processes_num, task_num=tasks_num, all_gradients=all_gradients, ms_list=ms_list)
    # p_optimal = tradeoff_solver(client_num=all_clients_num, task_num=tasks_num, all_gradients=all_gradients, active_num=sample_num, dis=d_is)

    p_s_i = p_optimal
    allocation_result = np.zeros(processes_num, dtype=int)
    for process_idx in range(processes_num):
        if abs(1 - np.sum(p_s_i[:, process_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, process_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num + 1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, process_idx]
        # ensure p_client sum to 1
        p_client = p_client / np.sum(p_client)
        allocation_result[process_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client)
    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_process_order = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    chosen_clients = [clients_process[i] for i in chosen_process_order]
    # get p_dict
    p_dict = []
    active_rate = len(chosen_clients) / processes_num

    for task_index in range(tasks_num):
        p_dict.append(
            [p_s_i[task_index][i] * client_task_ability[clients_process[i]] for i in range(processes_num) if
             allocation_result[i] == task_index])

    # store p_s_i, process_gradients_sumTasks
    if args.optimal_sampling is True:
        type = 'OS'
    elif args.approx_optimal is True:
        type = 'AS'
    else:
        type = 'none'
    file_path = save_path + 'psi_' + type + '.pkl'
    append_to_pickle(file_path, p_s_i)

    # record gradient_record
    file_path = save_path + 'gradient_record_' + type + '.pkl'
    append_to_pickle(file_path, gradient_record)

    # compute the punishment
    punishment_list = []
    for task_index in range(tasks_num):
        punishment_each_task = 0
        for process_index in range(processes_num):
            client_index = clients_process[process_index]
            if p_s_i[task_index][process_index] == 0:
                continue
            else:
                # if client_idx is active, then add it
                if client_index in chosen_clients:
                    punishment_each_task += dis[task_index][client_index] / client_task_ability[client_index] / \
                                            p_s_i[task_index][process_index]
        punishment_each_task = (punishment_each_task - 1) ** 2
        punishment_list.append(punishment_each_task)

    # save the punishment
    file_path = save_path + 'punishment_' + type + '.pkl'
    append_to_pickle(file_path, punishment_list)

    return clients_task, p_dict, chosen_clients



def aggregation_fair(loss_af_aggregation, loss_bf_aggregation):
    loss_diff = loss_af_aggregation - loss_bf_aggregation
    # we want loss_diff to be small
    # if loss_diff is negative, change to 0
    loss_diff = np.maximum(loss_diff, 1e-6)
    return loss_diff



def get_optimal_b(new_updates, old_updates, tasknum, clientnum, args):
    optimal_b_array = np.zeros((tasknum, clientnum))
    for task_index in range(tasknum):
        for client_index in range(clientnum):
            # get the optimal b, b = weight_product(new,old)/weight_product(old,old)
            if weight_norm(old_updates[task_index][client_index]) == 0:
                optimal_b = 0
                optimal_b_array[task_index][client_index] = optimal_b
            else:
                if args.lastLayer is True:
                    optimal_b = weight_product_lastLayer(new_updates[task_index][client_index],
                                               old_updates[task_index][client_index]) / weight_product_lastLayer(
                        old_updates[task_index][client_index], old_updates[task_index][client_index])
                else:
                    optimal_b = weight_product(new_updates[task_index][client_index], old_updates[task_index][client_index]) / weight_product(old_updates[task_index][client_index], old_updates[task_index][client_index])
                optimal_b_array[task_index][client_index] = optimal_b
    return optimal_b_array

def fixed_distribution(round, round_scale):
    # notice here we assume each client can only do one task.
    # the first 20 clients use round_robin in a manner.
    # the last 100 clients randomly choose a task to do
    order = round % round_scale # can be 0,1,2,3,4
    chosen_clients = []
    steps = int(20 // round_scale)
    for i in range(steps):
        chosen_clients.append(int(i*round_scale + order))
    p_dict = []
    p_dict.append([1/round_scale for i in range(steps)])
    # for clients beyond 20, randomly choose
    active_rate = 0.1
    active_num = int(100 * active_rate)
    chosen_clients += random.sample(range(20, 100), active_num)
    p_dict[0].extend([0.1 for i in range(active_num)])
    # only train the first task
    clients_task = [0] * len(chosen_clients)
    return clients_task, p_dict, chosen_clients


import math


def find_recent_allocation(allocation_record, task_index, client_index):
    current_total_round = len(allocation_record)
    for i in range(current_total_round-2, -1, -1): # reverse order
        if client_index in allocation_record[i].keys():
            if task_index in allocation_record[i][client_index]:
                return current_total_round - i - 1
    return -1 # the first time allocation


def find_recent_allocation_withP(allocation_record, task_index, client_index, psi):
    current_total_round = len(allocation_record)
    for i in range(current_total_round-2, -1, -1): # reverse order
        if client_index in allocation_record[i].keys():
            if task_index in allocation_record[i][client_index]:
                return current_total_round - i - 1, psi[i][task_index][client_index]
    return -1, 1 # the first time allocation


def approximate_decayb(new_updates, old_updates, tasknum, clientnum, allocation_record, chosen_clients, client_task, b0, decay_rates):
    # allocation_record include the current round allocation
    if len(allocation_record) == 1:
        return np.zeros((tasknum, clientnum))

    for i in range(len(chosen_clients)):
        task_index = client_task[i]
        client_index = chosen_clients[i]
        optimal_b_nom = weight_product(new_updates[task_index][client_index],
                                   old_updates[task_index][client_index])
        optimal_b_dom = weight_product(old_updates[task_index][client_index], old_updates[task_index][client_index])
        if optimal_b_dom == 0:
            optimal_b = b0
        else:
            optimal_b = optimal_b_nom / optimal_b_dom
        delta_t = find_recent_allocation(allocation_record, task_index, client_index)
        if delta_t == -1:
            decay = 0.0  # first time active
            decay_rates[task_index][client_index] = decay
        else:
            # linear decay
            decay = (b0 - optimal_b) / delta_t
            decay_rates[task_index][client_index] = decay
    return decay_rates


def average_beta(new_updates, old_updates, tasknum, chosen_clients, client_task):
    # allocation_record include the current round allocation
    average_beta = [0.0 for _ in range(tasknum)]
    task_count = [0 for _ in range(tasknum)]
    optimal_betas = []

    for i in range(len(chosen_clients)):
        task_index = client_task[i]
        client_index = chosen_clients[i]
        optimal_b_nom = weight_product(new_updates[task_index][client_index],
                                   old_updates[task_index][client_index])
        optimal_b_dom = weight_product(
            old_updates[task_index][client_index], old_updates[task_index][client_index])
        if optimal_b_dom == 0:
            optimal_b = 0
        else:
            optimal_b = optimal_b_nom / optimal_b_dom
        optimal_betas.append(optimal_b)
        average_beta[task_index] += optimal_b
        task_count[task_index] += 1
    for i in range(tasknum):
        if task_count[i] == 0:
            average_beta[i] = 0.8
        else:
            average_beta[i] = average_beta[i] / task_count[i]

    return average_beta, optimal_betas

def get_one_optimal_b(new_updates, old_updates):
    if weight_norm(old_updates) == 0:
        optimal_b = 0
    else:
        optimal_b = weight_product(new_updates, old_updates) / weight_product(old_updates, old_updates)
    return optimal_b


def get_optimal_distribution(m, dis, gradient_record, client_task_ability, clients_process, venn_matrix): # gradient record is norm
    # gradient_record: the shape is [task_index][client_index]
    # chosen_clients provide the index of the chosen clients in a random order
    # clients_task has the same order as chosen_clients
    # multiple tasks sampling will degenerate to single task sampling when task=1
    # therefore we can use the same function.
    sample_num = m  # m in the paper
    tasks_num = len(gradient_record)
    # random.shuffle(task_indices) # make task order random
    all_clients_num = len(gradient_record[0])
    processes_num = sum(client_task_ability) # client_task_ability [4,2,1,..] client 1 has 4 processes, client 2 has 2 processes

    all_gradients = np.zeros((tasks_num, processes_num))


    for task_index in range(tasks_num):
        for process_index in range(processes_num):
        # from U to U~ in the paper
            client_index = clients_process[process_index]
            all_gradients[task_index][process_index] = gradient_record[task_index][client_index] * dis[task_index][client_index] / client_task_ability[client_index] * venn_matrix[task_index][client_index]
    process_gradients_sumTasks = np.zeros(processes_num) # this is M_i in the proof
    for process_index in range(processes_num):
        for task_index in range(tasks_num):
            process_gradients_sumTasks[process_index] += all_gradients[task_index][process_index]

    # sort the gradients of the clients for this task, get a list of indices
    sorted_indices = np.argsort(process_gradients_sumTasks)

    n = processes_num
    m = sample_num

    l = n - m + 1
    best_l = l
    if m == 0: # if m=0, we get best_l = n+1 above, which is wrong. how to solve?
        best_l = n

    while True:
        l += 1
        if l > n:
            break
        # sum the first l smallest gradients
        sum_upto_l = sum(process_gradients_sumTasks[sorted_indices[i]] for i in range(l))
        upper = sum_upto_l / process_gradients_sumTasks[sorted_indices[l-1]]
        # if 0<m+l-n<=upper, then this l is good. find the largest l satisfying this condition
        if 0 < m + l - n <= upper:
            best_l = l
    # compute p
    p_s_i = np.zeros((tasks_num, processes_num))
    sum_upto_l = sum(process_gradients_sumTasks[sorted_indices[i]] for i in range(best_l))
    # print('sum_upto_l', sum_upto_l)
    for i in range(len(sorted_indices)):
        if i >= best_l:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = all_gradients[task_index][sorted_indices[i]] / process_gradients_sumTasks[sorted_indices[i]]
        else:
            for task_index in range(tasks_num):
                p_s_i[task_index][sorted_indices[i]] = (m + best_l - n) * all_gradients[task_index][sorted_indices[i]] / sum_upto_l
    # if nan appears, then set it to 0
    p_s_i[np.isnan(p_s_i)] = m/n
    return p_s_i



def get_one_optimal_b_ALT(new_updates, old_updates, p, q):
    if weight_norm(old_updates) == 0:
        optimal_b = 0
    else:
        if (p < 1e-6) or (q < 1e-6):  # if p is 0
            x = 1
            k = 0
        else:
            x = (1 - p) / p
            k = (1 - q) / q
        nom = weight_product(new_updates, old_updates)
        denom = weight_product(old_updates, old_updates)
        optimal_b = nom * x / (denom * (x + k))
    return optimal_b


def compute_p_active_once(psi, window_size):
    # psi is a list
    p_active_once = np.ones_like(psi[-1]) # tasknum, clientnum
    current_round = len(psi)
    if current_round < (window_size+1):
        return p_active_once
    else:
        for t_ in range(window_size):
            p_active_once *= (1 - psi[-2-t_])
        p_active_once = 1 - p_active_once
        # set 0 to 1 for elements in p_active_once
        p_active_once[p_active_once == 0.0] = 1.0
        p_active_once[p_active_once < 0.5] = 0.5
        return p_active_once


def alt_min(round, task_type, chosen_clients, client_task_ability,
            clients_process, venn_matrix, dis, gradient_new, gradient_old, all_weights_diff, args, save_path):
    # optimal beta closed-form:
    # beta = (h*G)^2 * (1/p-1) /(\|h\|^2 * (1/p + 1/q -2))
    # p=optimal_sampling(\|G-beta*h\|^2)
    client_num = len(gradient_new[0])
    task_num = len(gradient_new)
    m = len(chosen_clients)
    processes_num = sum(client_task_ability)
    iteration = 20

    betaH = copy.deepcopy(gradient_old)

    # initialize beta as all ones
    beta = np.ones((task_num, client_num)) * 0.5
    norms_array = all_weights_diff
    # initialize qsi

    if round == 0:
        psi = get_optimal_distribution(m=m, dis=dis, gradient_record=norms_array,
                                       client_task_ability=client_task_ability,
                                       clients_process=clients_process, venn_matrix=venn_matrix)
    else:
        psi_list_file = save_path + 'psi_OS.pkl'
        with open(psi_list_file, "rb") as f:
            psi_history = pickle.load(f)
        qsi = compute_p_active_once(psi_history, args.window_size)
        # initialize psi
        psi = np.ones((task_num, processes_num))
        for _ in range(iteration):
            pre_beta = beta
            pre_psi = psi
            # ------------------P step------------------
            # ------update \|G-beta*h\|------
            for task in range(task_num):
                LR = optimizer_config(task_type[task])
                for cl in range(client_num):
                    betaH[task][cl] = newupdate(weights=gradient_old[task][cl], b0=beta[task][cl])
                    norms_array[task][cl], _ = get_gradient_norm(
                        weights_this_round=gradient_new[task][cl],
                        weights_next_round=betaH[task][cl],
                        lr=LR)

            # ------compute norm (without learning rate)------
            psi = get_optimal_distribution(m=m, dis=dis, gradient_record=norms_array, client_task_ability=client_task_ability,
                                           clients_process=clients_process, venn_matrix=venn_matrix)
            # ------------------beta step------------------
            for task in range(task_num):
                for cl in range(client_num):
                    beta[task][cl] = get_one_optimal_b_ALT(new_updates=gradient_new[task][cl], old_updates=betaH[task][cl], p=psi[task][cl], q=qsi[task][cl])

            # ------------------check if converge------------------
            # print(f"beta error with iteration {beta}")
            # print(f"p error with iteration {psi[1][1]}")
    return beta, psi

def sampling_distribution(p_s_i, tasks_num, clients_process, client_task_ability, save_path, args):
    all_clients_num = args.num_clients
    processes_num = sum(client_task_ability)

    allocation_result = np.zeros(processes_num, dtype=int)
    for process_idx in range(processes_num):
        if abs(1-np.sum(p_s_i[:, process_idx])) < 1e-6:
            p_not_choose = 0
        else:
            p_not_choose = 1 - np.sum(p_s_i[:, process_idx])
        # append p_not_choose to the head of p_s_i
        p_client = np.zeros(tasks_num+1)
        p_client[0] = p_not_choose
        p_client[1:] = p_s_i[:, process_idx]
        allocation_result[process_idx] = np.random.choice(np.arange(-1, tasks_num), p=p_client) # appear NaN
    allocation_result = allocation_result.tolist()
    clients_task = [s for s in allocation_result if s != -1]
    chosen_process_order = [i for i in range(len(allocation_result)) if allocation_result[i] != -1]
    chosen_clients = [clients_process[i] for i in chosen_process_order]
    # get p_dict
    p_dict = []
    active_rate = len(chosen_clients)/all_clients_num
    for task_index in range(tasks_num):
        p_dict.append([p_s_i[task_index][i] * client_task_ability[clients_process[i]] for i in range(processes_num) if allocation_result[i] == task_index])

    # store p_s_i, process_gradients_sumTasks
    if args.optimal_sampling is True:
        type = 'OS'
    elif args.approx_optimal is True:
        type = 'AS'
    else:
        type = 'none'
    file_path = save_path + 'psi_'+type+'.pkl'
    append_to_pickle(file_path, p_s_i)

    return clients_task, p_dict, chosen_clients


def generate_given_psi(givenProb, N, active_rate, rounds):
    p_high = 1 + givenProb
    p_low = 1
    clients_num = N
    active_num = int(N * active_rate)
    # create distribution, half are p_high, half are p_low
    p_all = [p_high for i in range(clients_num)]
    p_all[clients_num // 2:] = [p_low for i in range(clients_num - clients_num // 2)]
    # normalize to probabilities
    p_all = np.array(p_all).reshape(1, -1)
    p_all = p_all / np.sum(p_all) * active_num
    psi = [p_all for _ in range(rounds)]

    return psi

def sample_unbalanced_distribution(clients_process, m, givenProb, task_num):
    # clients_process: order of clients, e.g., [0,1,2,3,4,5,6,7,8,9] (total 9 clients)
    # m: allowed communication (int)
    # givenProb: unbalanced coefficient alpha. half client get 1+alpha, half get 1, then normalize to probabilities
    # return chosen_clients: list of chosen clients, p_all: list of p for each client
    p_high = 1 + givenProb
    p_low = 1
    clients_num = len(clients_process)
    active_num = m
    # create distribution, half are p_high, half are p_low
    p_all = [p_high for i in range(clients_num)]
    p_all[clients_num//2:] = [p_low for i in range(clients_num - clients_num//2)]
    # normalize to probabilities
    p_all = np.array(p_all)
    p_all = p_all / np.sum(p_all) * active_num
    # sample clients
    chosen_clients = []
    allocation_result = np.zeros(clients_num, dtype=int)
    for idx in range(clients_num):
        p = p_all[idx]
        # binomial sampling, decide 0 or 1
        allocation_result[idx] = np.random.choice([0, 1], p=[1.0 - p, p])
        # convert allocation_result to list of indices (selected_clients)
    for idx in range(clients_num):
        if allocation_result[idx] == 1:
            chosen_clients.append(idx)
    # create p_list
    p_list = [[] for i in range(task_num)]
    clients_task = []
    for client in chosen_clients:
        # decide which task to do (random sampling)
        clients_task.append(np.random.choice(task_num))
        # record the probability in p_list
        p_list[clients_task[-1]].append(p_all[clients_process.index(client)])
    return chosen_clients, clients_task, p_list  # here we don't consider tasks

