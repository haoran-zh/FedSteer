import torch
import numpy as np
import utility.optimal_sampling as optimal_sampling
import copy
import pickle

def federated(models_state_dict, local_data_nums, aggregation_mtd, numUsersSel):

    global_state_dict = models_state_dict[0].copy()
    global_keys = list(global_state_dict.keys())

    for key in global_keys:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    # Sum the state_dicts of all client models
    for i, model_state_dict in enumerate(models_state_dict):
        for key in global_keys:
            if aggregation_mtd=='pkOverSumPk':
                global_state_dict[key] += local_data_nums[i]/np.sum(local_data_nums)  * model_state_dict[key]
                # W_global = sum_
            elif aggregation_mtd=='numUsersInv':
                global_state_dict[key] += 1/len(local_data_nums)  * model_state_dict[key]
                # the above means 1/ num users
            #global_state_dict[key] += local_data_nums[i]/np.sum(local_data_nums)  * model_state_dict[key]

    return global_state_dict

def federated_prob(global_weights, models_gradient_dict, local_data_num, p_list, args, chosen_clients, tasks_local_training_loss, lr):

    global_weights_dict = global_weights.state_dict()
    global_keys = list(global_weights_dict.keys())
    # Sum the state_dicts of all client models
    # sum loss power a-1
    alpha = args.alpha
    N = args.num_clients
    L = 1
    dis_s = local_data_num
    denominator = 0
    # aggregate
    if (args.fairness == 'notfair'):
        denominator = 1

        for i, gradient_dict in enumerate(models_gradient_dict):
            d_i = dis_s[chosen_clients[i]]
            for key in global_keys:
                global_weights_dict[key] -= (d_i / p_list[i]) * gradient_dict[key] / denominator
    elif args.fairness == 'taskfair':
        # get f_s and max gradient (H_s)
        f_s = 0
        H_s = 0
        assert len(tasks_local_training_loss) == N
        for i in range(N):
            d_is = dis_s[i]
            f_s += tasks_local_training_loss[i] * d_is
        for i, gradient_dict in enumerate(models_gradient_dict):
            d_is = dis_s[chosen_clients[i]]
            norm = sum(torch.norm(diff, p=2) ** 2 for diff in gradient_dict.values()) ** 0.5 * L
            if H_s < norm*d_is:
                H_s = norm*d_is
        denominator = (alpha-1) * (N * H_s)**2 + f_s * L
        #print(denominator)
        #print(f_s)
        for i, gradient_dict in enumerate(models_gradient_dict):
            d_is = dis_s[chosen_clients[i]]
            for key in global_keys:
                global_weights_dict[key] -= d_is / p_list[i] * f_s * gradient_dict[key]*L / denominator
    else:
        print("aggregation wrong!")
        exit(1)

    return global_weights_dict

def compute_p_active_once(psi, window_size, args):
    # psi is a list
    p_active_once = np.ones_like(psi[-1]) # tasknum, clientnum
    current_round = len(psi)
    if current_round < (window_size+1):
        return p_active_once, -1
    else:
        for t_ in range(window_size):
            p_active_once *= (1 - psi[-2-t_])
        p_active_once = 1 - p_active_once
        # set 0 to 1 for elements in p_active_once
        p_active_once[p_active_once == 0.0] = 1.0
        # set any value in p_active_once to no less than 0.5
        # record how many clients are below the LB
        num_clients_below_LB = np.sum(p_active_once < args.LB)
        p_active_once[p_active_once < args.LB] = args.LB
        return p_active_once, num_clients_below_LB


def compute_p_active_once_fullfill(psi, window_size):
    # psi is a list
    p_active_once = np.ones_like(psi[-1]) # tasknum, clientnum
    current_round = len(psi)
    if current_round < (window_size+1):
        # fill the first few rounds with p[0]
        for t_ in range(window_size):
            if t_ < (current_round-1):
                p_active_once *= (1 - psi[-2-t_])
            else:
                p_active_once *= (1 - psi[0])
        p_active_once = 1 - p_active_once
        # set 0 to 1 for elements in p_active_once
        p_active_once[p_active_once == 0.0] = 1.0
        return p_active_once
    else:
        for t_ in range(window_size):
            p_active_once *= (1 - psi[-2-t_])
        p_active_once = 1 - p_active_once
        # set 0 to 1 for elements in p_active_once
        p_active_once[p_active_once == 0.0] = 1.0
        return p_active_once



def window_states(allocation_history, task_index, args):
    # use this function at the start, create a list to include all clients within the bound
    clients_within_window = {}
    clients_num = args.num_clients
    for i in range(clients_num):
        delta_t = optimal_sampling.find_recent_allocation(allocation_history, task_index, i)
        if delta_t <= args.window_size:
            clients_within_window[i] = delta_t
    window_max = args.window_max
    # if clients_within_window includes clients more than window_max, remove clients with largest delta_t
    if len(clients_within_window) > window_max:
        clients_within_window = dict(sorted(clients_within_window.items(), key=lambda item: item[1])[:window_max])
    return clients_within_window

def window_states_Krank(allocation_history, task_index, args):
    # use this function at the start, create a list to include all clients within the bound
    # if Krank is True, then window_max (K) means the most recent K stale updates
    # and window_size will not be used
    clients_recent_activeTime = {}
    K = args.window_max
    clients_num = args.num_clients
    for i in range(clients_num):
        delta_t = optimal_sampling.find_recent_allocation(allocation_history, task_index, i)
        clients_recent_activeTime[i] = delta_t
    # rank clients by their active time
    clients_within_window = dict(sorted(clients_recent_activeTime.items(), key=lambda item: item[1])[:K])
    # return the most recent K clients
    return clients_within_window



def federated_stale(global_weights, models_gradient_dict, local_data_num, p_list, args, chosen_clients, old_global_weights, decay_beta, allocation_result, task_index, save_path):
    global_weights_dict = global_weights.state_dict()
    global_keys = list(global_weights_dict.keys())
    # Sum the state_dicts of all client models
    # sum loss power a-1
    alpha = args.alpha
    N = args.num_clients
    L = 1
    dis_s = local_data_num
    total_rounds = len(allocation_result)

    if args.MILA is True:  # no problem, because MIFA doesn't have optimal sampling, so we get original h_i without beta here
        clients_num = len(dis_s)
        for i in range(clients_num):
            if i not in chosen_clients:
                for key in global_keys:
                    global_weights_dict[key] -= dis_s[i] * old_global_weights[i][key]
            else:
                for key in global_keys:
                    global_weights_dict[key] -= dis_s[i] * models_gradient_dict[chosen_clients.index(i)][key]
    else:
        # other method, FedVARP, FedStale, our methods
        # For FedVARP(args.optimal_sampling is False),
        # decay_beta_record is always totally 0, old_global_weights is the original
        # FedStale has args.skipOS as True.
        # dict, key is client index, value is delta_t
        if args.Krank is True:
            clients_within_window = window_states_Krank(allocation_result, task_index, args)
        else:
            clients_within_window = window_states(allocation_result, task_index, args)


        if args.ubwindow is True:
            if args.givenProb != 0.0:
                # generate psi based on given probability
                psi = optimal_sampling.generate_given_psi(args.givenProb, N, active_rate=args.C, rounds=total_rounds)  # we don't consider multiple processors here
            else:
                # read past probabilities, divide probability to ensure unbiasedness
                psi_list_file = save_path + 'psi_OS.pkl'
                with open(psi_list_file, "rb") as f:
                    psi = pickle.load(f)
            # compute probability of being active at least once in the window
            p_active_once, num_clients_below_LB = compute_p_active_once(psi, args.window_size, args)
            # store num_clients_below_LB
            num_clients_below_LB_file = save_path + 'num_clients_below_LB.pkl'
            optimal_sampling.append_to_pickle(num_clients_below_LB_file, num_clients_below_LB)

            for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
                d_i = dis_s[chosen_clients[i]]
                h_i = old_global_weights[chosen_clients[i]]
                for key in global_keys:
                    global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key])


            # only include clients within the window
            clients_num = len(dis_s)

            for i in range(clients_num):
                # to decide if client i is within the window
                # delta_t = optimal_sampling.find_recent_allocation(allocation_result, task_index, i)
                # use clients_within_window, keys: client index, values: delta_t
                if i in clients_within_window:
                    d_i = dis_s[i]
                    d_i = d_i / p_active_once[task_index, i]  # where we make it unbiased
                    h_i = old_global_weights[i]
                    for key in global_keys:
                        global_weights_dict[key] -= d_i * h_i[key]

        elif args.ubwindow2 is True:
            # read past probabilities, divide probability to ensure unbiasedness
            for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
                # delta_t = optimal_sampling.find_recent_allocation(allocation_result, task_index, chosen_clients[i])
                if chosen_clients[i] in clients_within_window:
                    d_i = dis_s[chosen_clients[i]]
                    h_i = old_global_weights[chosen_clients[i]]
                    for key in global_keys:
                        global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key])
                else:
                    d_i = dis_s[chosen_clients[i]]
                    for key in global_keys:
                        global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key])

            # only include clients within the window
            clients_num = len(dis_s)

            for i in range(clients_num):
                # to decide if client i is within the window
                #delta_t = optimal_sampling.find_recent_allocation(allocation_result, task_index, i)
                if i in clients_within_window:
                    d_i = dis_s[i]
                    h_i = old_global_weights[i]
                    for key in global_keys:
                        global_weights_dict[key] -= d_i * h_i[key]

        elif args.ubwindow3 is True:
            # read past probabilities, divide probability to ensure unbiasedness
            psi_list_file = save_path + 'psi_OS.pkl'
            with open(psi_list_file, "rb") as f:
                psi = pickle.load(f)
            # compute probability of being active at least once in the window
            p_active_once, num_clients_below_LB = compute_p_active_once(psi, args.window_size, args)
            # store num_clients_below_LB
            num_clients_below_LB_file = save_path + 'num_clients_below_LB.pkl'
            optimal_sampling.append_to_pickle(num_clients_below_LB_file, num_clients_below_LB)

            for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
                #delta_t = optimal_sampling.find_recent_allocation(allocation_result, task_index, chosen_clients[i])
                if chosen_clients[i] in clients_within_window:
                    d_i = dis_s[chosen_clients[i]]
                    h_i = old_global_weights[chosen_clients[i]]
                    for key in global_keys:
                        global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key]/p_active_once[task_index, chosen_clients[i]])
                else:
                    d_i = dis_s[chosen_clients[i]]
                    for key in global_keys:
                        global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key])


            # only include clients within the window
            clients_num = len(dis_s)

            for i in range(clients_num):
                # to decide if client i is within the window
                #delta_t = optimal_sampling.find_recent_allocation(allocation_result, task_index, i)
                if i in clients_within_window:
                    d_i = dis_s[i]
                    d_i = d_i / p_active_once[task_index, i]
                    h_i = old_global_weights[i]
                    for key in global_keys:
                        global_weights_dict[key] -= d_i * h_i[key]
        elif args.window is True:  # biased window
            for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
                d_i = dis_s[chosen_clients[i]]
                h_i = old_global_weights[chosen_clients[i]]
                for key in global_keys:
                    global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key])


            # only include clients within the window
            clients_num = len(dis_s)

            for i in range(clients_num):
                # to decide if client i is within the window
                # delta_t = optimal_sampling.find_recent_allocation(allocation_result, task_index, i)
                if i in clients_within_window:
                    d_i = dis_s[i]
                    h_i = old_global_weights[i]
                    for key in global_keys:
                        global_weights_dict[key] -= d_i * h_i[key]

        else: # sum for all clients
            for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
                d_i = dis_s[chosen_clients[i]]
                h_i = old_global_weights[chosen_clients[i]]
                for key in global_keys:
                    # if we use summation window, then should minus h_i * window_size
                    global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key])
            clients_num = len(dis_s)
            for i in range(clients_num):
                d_i = dis_s[i]
                h_i = old_global_weights[i]
                for key in global_keys:
                    global_weights_dict[key] -= d_i * h_i[key]
    return global_weights_dict