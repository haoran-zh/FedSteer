import torch
import numpy as np
import utility.optimal_sampling as optimal_sampling
import copy
import pickle
from utility.optimal_sampling import weight_minus, weight_add
from utility.matching import OMP

def candidates_normalization(candidates):
    # compute the avg norm of candidates
    # rescale candidates norm to the avg norm
    # candidates is a list of tensors
    candidate_norms = []
    normalized_candidates = []
    for c in candidates:
        norm = torch.norm(c, p=2)
        candidate_norms.append(norm.item())
    # compute the average of candidate norms
    avg_norm = np.mean(candidate_norms)
    # rescale each candidate to the avg norm
    for i in range(len(candidates)):
        if candidate_norms[i] == 0:
            normalized_candidates.append(candidates[i])
        else:
            normalized_candidates.append(candidates[i] * (avg_norm / candidate_norms[i]))
    return normalized_candidates

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


def window_states_optimalV(allocation_history, task_index, args, window_size):
    # use this function at the start, create a list to include all clients within the bound
    clients_within_window = {}
    clients_num = args.num_clients
    for i in range(clients_num):
        delta_t = optimal_sampling.find_recent_allocation(allocation_history, task_index, i)
        if delta_t <= window_size:
            clients_within_window[i] = delta_t
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

def Krank_active(allocation_history, task_index, args, K):
    # use this function at the start, create a list to include all clients within the bound
    # if Krank is True, then window_max (K) means the most recent K stale updates
    # and window_size will not be used
    clients_recent_activeTime = {}
    clients_num = args.num_clients
    for i in range(clients_num):
        delta_t = optimal_sampling.find_recent_allocation(allocation_history, task_index, i)
        clients_recent_activeTime[i] = delta_t
    # rank clients by their active time
    clients_within_window = dict(sorted(clients_recent_activeTime.items(), key=lambda item: item[1])[:K])
    # return the most recent K clients
    return clients_within_window


def updateV(H, models_gradient_dict, clients_within, args):
    """
    H: list of dictionaries representing q_i^t for each client i.
    models_gradient_dict: list of dictionaries representing G_i^t for each client i.

    Returns:
      s: a numpy array of shape (N,N) with the optimal coefficients s_{ij}.
      Q: a list of dictionaries, where Q[i] = sum_j s_{ij} * (flattened difference corresponding to H[j]-H[i])
         (unflattened to have same structure as H[i]).
    """

    # helper function to flatten a dict of tensors into a 1d tensor.
    # We sort the keys so that the order is deterministic.
    def flatten_weights(weights):
        return torch.cat([weights[k].reshape(-1) for k in sorted(weights.keys())])

    # helper function to unflatten a flat tensor back into the dict shape given by template.
    def unflatten_weights(flat_tensor, template):
        new_weights = {}
        pointer = 0
        for k in sorted(template.keys()):
            shape = template[k].shape
            numel = template[k].numel()
            new_weights[k] = flat_tensor[pointer:pointer + numel].view(shape)
            pointer += numel
        return new_weights

    def safe_solve(A, b, epsilon=1e-6):
            return torch.linalg.pinv(A) @ b

    #clients_within = list(clients_within.keys())
    # clients_within = [i for i in range(40)]  # include all clients
    N = len(H)


    # First, build V_list and r_list where each V_list[i] is a matrix (d x N) and each r_list[i] is a vector (d,)
    V_list = []
    r_list = []

    # Determine the dimension by flattening one of the H dictionaries
    H_reduced = [H[client] for client in clients_within]
    d = flatten_weights(H_reduced[0]).shape[0]
    N_reduced = len(clients_within)

    force_range = args.force_range

    for i in range(N):
        # Compute r_i = flatten( G_i^t - H[i] ).
        # We use the provided weight_minus function, assumed to work as:
        #    weight_minus(A, B) returns a dict with (A[k]-B[k]) for each key k.
        r_i = flatten_weights(weight_minus(models_gradient_dict[i], H[i]))
        r_list.append(r_i)

        # Build V_i: for each j, if j==i then a zero vector, else flatten( H[j]-H[i] )
        V_i_cols = []
        for j in clients_within:
            if i == j:
                v_ij = torch.zeros(d)
            else:
                minus_result = weight_minus(H[j], H[i])
                v_ij = flatten_weights(minus_result)
                # rescale it to 1/N_reduced
                v_ij = v_ij / N_reduced
            # Make it a column vector
            V_i_cols.append(v_ij.unsqueeze(1))  # shape (d,1)
        # Stack columns to form V_i (shape d x N)
        # do the normalization for V_i_cols
        if args.norm_candidates is True:
            V_i_cols = candidates_normalization(V_i_cols)
        V_i = torch.cat(V_i_cols, dim=1)
        V_list.append(V_i)

    # Now solve for s for each i and compute Q.
    s = np.zeros((N, N_reduced))  # each row i is the coefficients s_i
    Q = []  # Q[i] will be unflattened back to dict format
    for i in range(N):
        V_i = V_list[i]  # shape (d, N)
        r_i = r_list[i]  # shape (d,)
        # Compute the normal equations: A s_i = b, with A = V_i^T V_i and b = V_i^T r_i.
        A = V_i.t() @ V_i  # shape (N, N)
        b = V_i.t() @ r_i  # shape (N,)
        # Solve for s_i; if A is not invertible, use a pseudo-inverse.
        s_i = safe_solve(A, b)
        # Store the coefficients
        s[i, :] = s_i.detach().cpu().numpy()
        # Compute Q[i] = V_i @ s_i, which is a flat vector of dimension d.
        if force_range:
            s_i = np.clip(s_i, 0.0, 1.0)
        Q_i_flat = V_i @ s_i  # shape (d,), the result of sV
        # Unflatten Q_i back to the dictionary structure of H[i].
        Q_i = unflatten_weights(Q_i_flat, H[i])
        # adjust to the form used in aggregation
        # Q_new = Q_i + H[i]
        Q_update = weight_add(Q_i, H[i])  # make sure it approximates the Gradients
        Q.append(Q_update)

    return s, Q

def get_minus_norm_square(weights_A, weights_B):
    # get gradient by subtracting weights_next_round from weights_this_round
    weight_diff = {name: (weights_A[name] - weights_B[name]).cpu() for name in weights_A}
    # Calculate the L2 norm of the weight differences
    # bound in case appear nan
    norm = sum(torch.norm(diff, p=2) ** 2 for diff in weight_diff.values())
    norm.item()
    if torch.isnan(norm):
        norm = torch.tensor(0.0)
    return norm.item()


def compute_variance_direct(G, di, p_all, stale_adjusted):
    client_num = len(stale_adjusted)
    var = 0.0
    for i in range(client_num):
        var += di[i]**2 / p_all[i] * get_minus_norm_square(G[i], stale_adjusted[i])
    return var


def updateV_direct(H, models_gradient_dict, clients_within, args):
    """
    H: list of dictionaries representing q_i^t for each client i.
    models_gradient_dict: list of dictionaries representing G_i^t for each client i.

    Returns:
      s: a numpy array of shape (N,N) with the optimal coefficients s_{ij}.
      Q: a list of dictionaries, where Q[i] = sum_j s_{ij} * (flattened difference corresponding to H[j]-H[i])
         (unflattened to have same structure as H[i]).
    """

    # helper function to flatten a dict of tensors into a 1d tensor.
    # We sort the keys so that the order is deterministic.
    def flatten_weights(weights):
        return torch.cat([weights[k].reshape(-1) for k in sorted(weights.keys())])

    def last_layer_flatten_weights(weights):
        # only flatten the last layer of the model
        last_layer_key = list(weights.keys())[-1]
        return weights[last_layer_key].reshape(-1)

    # helper function to unflatten a flat tensor back into the dict shape given by template.
    def unflatten_weights(flat_tensor, template):
        new_weights = {}
        pointer = 0
        for k in sorted(template.keys()):
            shape = template[k].shape
            numel = template[k].numel()
            new_weights[k] = flat_tensor[pointer:pointer + numel].view(shape)
            pointer += numel
        return new_weights

    def safe_solve(A, b, epsilon=1e-6):
        return torch.linalg.pinv(A) @ b

    # clients_within = list(clients_within.keys())
    # clients_within = [i for i in range(40)]  # include all clients
    N = len(H)


    # First, build V_list and r_list where each V_list[i] is a matrix (d x N) and each r_list[i] is a vector (d,)
    V_list = []  # Q actually, stale
    VLL_list = []
    r_list = []  # new gradient
    rLL_list = []

    # Determine the dimension by flattening one of the H dictionaries
    H_reduced = [H[client] for client in clients_within]
    d = flatten_weights(H_reduced[0]).shape[0]
    N_reduced = len(clients_within)

    force_range = args.force_range

    for i in range(N):
        if args.lastLayer is True:
            rLL_i = last_layer_flatten_weights(models_gradient_dict[i])
            rLL_list.append(rLL_i)
        r_i = flatten_weights(models_gradient_dict[i])
        r_list.append(r_i)

        # Build V_i: for each j, if j==i then a zero vector, else flatten( H[j]-H[i] )
        V_i_cols = []
        VLL_i_cols = []
        for j in clients_within:
            if args.lastLayer is True:
                vLL_ij = last_layer_flatten_weights(H[j])
                vLL_ij = vLL_ij / N_reduced
                VLL_i_cols.append(vLL_ij.unsqueeze(1))
            v_ij = flatten_weights(H[j])
            # rescale it to 1/N_reduced
            v_ij = v_ij / N_reduced
            # Make it a column vector
            V_i_cols.append(v_ij.unsqueeze(1))  # shape (d,1)
        # Stack columns to form V_i (shape d x N)
        if args.lastLayer is True:
            VLL_i = torch.cat(VLL_i_cols, dim=1)
            VLL_list.append(VLL_i)
        V_i = torch.cat(V_i_cols, dim=1)
        V_list.append(V_i)

    # Now solve for s for each i and compute Q.
    s = np.zeros((N, N_reduced))  # each row i is the coefficients s_i
    Q = []  # Q[i] will be unflattened back to dict format

    if args.lastLayer is True:
        V_adpt_list = VLL_list
        r_adpt_list = rLL_list
    else:
        V_adpt_list = V_list
        r_adpt_list = r_list

    for i in range(N):
        V_i = V_adpt_list[i]  # shape (d, N)  V_i is the same as Q_i, the stale gradients
        r_i = r_adpt_list[i]  # shape (d,)  r_i is the new gradients
        # Compute the normal equations: A s_i = b, with A = V_i^T V_i and b = V_i^T r_i.
        A = V_i.t() @ V_i  # shape (N, N)
        b = V_i.t() @ r_i  # shape (N,)
        # Solve for s_i; if A is not invertible, use a pseudo-inverse.
        s_i = safe_solve(A, b)
        # Store the coefficients
        s[i, :] = s_i.detach().cpu().numpy()
        # Compute Q[i] = V_i @ s_i, which is a flat vector of dimension d.
        if force_range:
            s_i = np.clip(s_i, 0.0, 1.0)
        V_all = V_list[i]
        Q_i_flat = V_all @ s_i  # shape (d,), the result of sV
        # Unflatten Q_i back to the dictionary structure of H[i].
        Q_i = unflatten_weights(Q_i_flat, H[i])
        Q_update = Q_i
        Q.append(Q_update)
    args.s_maintain = copy.deepcopy(s)
    return s, Q


def updateV_direct_maintain(H, models_gradient_dict, clients_within, args):
    """
    H: list of dictionaries representing q_i^t for each client i.
    models_gradient_dict: list of dictionaries representing G_i^t for each client i.

    Returns:
      s: a numpy array of shape (N,N) with the optimal coefficients s_{ij}.
      Q: a list of dictionaries, where Q[i] = sum_j s_{ij} * (flattened difference corresponding to H[j]-H[i])
         (unflattened to have same structure as H[i]).
    """

    # helper function to flatten a dict of tensors into a 1d tensor.
    # We sort the keys so that the order is deterministic.
    def flatten_weights(weights):
        return torch.cat([weights[k].reshape(-1) for k in sorted(weights.keys())])

    # helper function to unflatten a flat tensor back into the dict shape given by template.
    def unflatten_weights(flat_tensor, template):
        new_weights = {}
        pointer = 0
        for k in sorted(template.keys()):
            shape = template[k].shape
            numel = template[k].numel()
            new_weights[k] = flat_tensor[pointer:pointer + numel].view(shape)
            pointer += numel
        return new_weights


    # clients_within = list(clients_within.keys())
    # clients_within = [i for i in range(40)]  # include all clients
    N = len(H)


    # First, build V_list and r_list where each V_list[i] is a matrix (d x N) and each r_list[i] is a vector (d,)
    V_list = []  # Q actually, stale
    r_list = []  # new gradient

    # Determine the dimension by flattening one of the H dictionaries
    N_reduced = len(clients_within)

    force_range = args.force_range

    for i in range(N):
        # Build V_i: for each j, if j==i then a zero vector, else flatten( H[j]-H[i] )
        V_i_cols = []
        for j in clients_within:
            v_ij = flatten_weights(H[j])
            # rescale it to 1/N_reduced
            v_ij = v_ij / N_reduced
            # Make it a column vector
            V_i_cols.append(v_ij.unsqueeze(1))  # shape (d,1)
        # Stack columns to form V_i (shape d x N)
        V_i = torch.cat(V_i_cols, dim=1)
        V_list.append(V_i)

    # Now solve for s for each i and compute Q.
    Q = []  # Q[i] will be unflattened back to dict format

    for i in range(N):
        s_i = args.s_maintain[i, :]  # use the maintained s_i
        # Compute Q[i] = V_i @ s_i, which is a flat vector of dimension d.
        if force_range:
            s_i = np.clip(s_i, 0.0, 1.0)
        V_all = V_list[i]
        Q_i_flat = V_all @ s_i  # shape (d,), the result of sV
        # Unflatten Q_i back to the dictionary structure of H[i].
        Q_i = unflatten_weights(Q_i_flat, H[i])
        Q_update = Q_i
        Q.append(Q_update)
    return Q



def federated_stale(global_weights, models_gradient_dict, local_data_num, p_list, args, chosen_clients, old_global_weights, old_global_weights_previous, decay_beta, allocation_result, task_index, save_path, allnew_gradients):
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
        elif args.optimalV is True:
            if args.randomK > 0:
                client_num = args.num_clients
                randomK = args.randomK
                # select randomK clients out of C
                randomK_clients = np.random.choice(client_num, randomK, replace=False)
                clients_within = randomK_clients
            elif args.randomKglobal > 0:
                # use clients with similar distribution
                # if first round, decide random, else, use previous random
                if total_rounds < 2:
                    client_num = args.num_clients
                    randomK = args.randomKglobal
                    # select randomK clients out of C
                    randomK_clients = np.random.choice(client_num, randomK, replace=False)
                    clients_within = randomK_clients
                    args.clients_within_global = clients_within
                else:
                    clients_within = args.clients_within_global
            else:
                client_num = int(args.num_clients)
                clients_within = [i for i in range(client_num)]
            if args.effV is True:
                s, Q = updateV(old_global_weights_previous, old_global_weights, clients_within, args)
            else:
                s, Q = updateV(old_global_weights, allnew_gradients, clients_within, args)

            # record optimal s
            s_file = save_path + 's.pkl'
            optimal_sampling.append_to_pickle(s_file, s)

            # record the variance values
            var = compute_variance_direct(G=allnew_gradients, di=dis_s, p_all=args.p_all, stale_adjusted=Q)
            var_file = save_path + 'var.pkl'
            optimal_sampling.append_to_pickle(var_file, var)

            for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
                d_i = dis_s[chosen_clients[i]]
                h_i = Q[chosen_clients[i]]
                for key in global_keys:
                    global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key])
            clients_num = len(dis_s)
            for i in range(clients_num):
                d_i = dis_s[i]
                h_i = Q[i]
                for key in global_keys:
                    global_weights_dict[key] -= d_i * h_i[key]
        elif args.V_direct is True:
            # random sample k clients as the base
            if args.OMP is True: # change OMP to make it only work on the first 5 rounds
                if total_rounds <= 5:
                    clients_within, s = OMP(stale_gradients=old_global_weights, fresh_gradients=allnew_gradients, d_list=dis_s,
                                        p_list=args.p_all, k=args.K, Lambda=args.lam, args=args)
                    args.clients_within_global = clients_within
                    args.s_maintain = copy.deepcopy(s)
                    # save clients_within_global
                    clientSet_file = save_path + 'cSet.pkl'
                    optimal_sampling.append_to_pickle(clientSet_file, clients_within)
                    Q = updateV_direct_maintain(old_global_weights, allnew_gradients, args.clients_within_global, args)
                else: # after round 5, use previous clients_within_global
                    clients_within = args.clients_within_global
                    # optimize optimal s
                    if total_rounds % args.s_slot == 0:
                        s, Q = updateV_direct(old_global_weights, allnew_gradients, clients_within, args)
                    else:
                        Q = updateV_direct_maintain(old_global_weights, allnew_gradients, clients_within, args)
                # Q = updateV_direct_maintain(old_global_weights, allnew_gradients, args.clients_within_global, args)


            else:
                if args.randomK > 0:
                    client_num = args.num_clients
                    randomK = args.randomK
                    # select randomK clients out of C
                    randomK_clients = np.random.choice(client_num, randomK, replace=False)
                    clients_within = randomK_clients
                    args.clients_within_global = randomK_clients
                elif args.randomKglobal > 0:
                    # use clients with similar distribution
                    # if first round, decide random, else, use previous random
                    if total_rounds < 2:
                        client_num = args.num_clients
                        randomK = args.randomKglobal
                        # select randomK clients out of C
                        randomK_clients = np.random.choice(client_num, randomK, replace=False)
                        clients_within = randomK_clients
                        args.clients_within_global = clients_within
                    else:
                        # use previous random clients
                        clients_within = args.clients_within_global
                elif args.recentK > 0:
                    # select most recent K active clients
                    clients_within = Krank_active(allocation_result, task_index, args, K=args.recentK)
                    if len(clients_within) < args.recentK:
                        client_num = int(args.num_clients)
                        clients_within = [i for i in range(client_num)]
                    args.clients_within_global = clients_within

                else:
                    client_num = int(args.num_clients)
                    clients_within = [i for i in range(client_num)]

                if args.effV is True:
                    if (total_rounds % args.s_slot == 0) or (total_rounds <= 5):
                        s, Q = updateV_direct(old_global_weights_previous, old_global_weights, clients_within, args)
                    else:
                        Q = updateV_direct_maintain(old_global_weights, allnew_gradients, clients_within, args)
                else:
                    if (total_rounds % args.s_slot == 0) or (total_rounds <= 5):
                        s, Q = updateV_direct(old_global_weights, allnew_gradients, clients_within, args)
                    else:
                        Q = updateV_direct_maintain(old_global_weights, allnew_gradients, clients_within, args)

            # record optimal s
            s_file = save_path + 's.pkl'
            optimal_sampling.append_to_pickle(s_file, args.s_maintain)

            # record the variance values
            var = compute_variance_direct(G=allnew_gradients, di=dis_s, p_all=args.p_all, stale_adjusted=Q)
            var_file = save_path + 'var.pkl'
            optimal_sampling.append_to_pickle(var_file, var)


            for i, gradient_dict in enumerate(models_gradient_dict):  # active clients
                d_i = dis_s[chosen_clients[i]]
                h_i = Q[chosen_clients[i]]
                for key in global_keys:
                    global_weights_dict[key] -= (d_i / p_list[i]) * (gradient_dict[key] - h_i[key])
            clients_num = len(dis_s)
            for i in range(clients_num):
                d_i = dis_s[i]
                h_i = Q[i]
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