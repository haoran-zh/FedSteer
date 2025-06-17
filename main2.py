import numpy as np
import torch
import utility.dataset as dataset
from utility.preprocessing import preprocessing
from utility.load_model import load_model
from utility.training import training, training_all, training_scaffold
from utility.evalation import evaluation, get_local_loss, group_fairness_evaluation, get_local_acc
from utility.aggregation import federated, federated_prob, federated_stale
from utility.taskallocation import get_task_idx, get_task_id_RR
from utility.config import optimizer_config
import copy
import random
import pickle
import time
import sys
import math
import os
from torch.utils.data import Subset
from tqdm import tqdm
import argparse
from utility.parser import ParserArgs
import utility.optimal_sampling as optimal_sampling


if __name__=="__main__":
    parser = ParserArgs()
    args = parser.get_args()
    exp_num = args.exp_num
    random_seed = args.seed  # default 13
    C = args.C  # communication active rate.
    num_clients = args.num_clients  # default 30 #100
    numUsersSel = C * num_clients # numUsersel

    normalization = 'accuracy'  # accuracy
    num_round = args.round_num  # 100#200
    print('num_round', num_round)
    local_epochs = args.local_epochs
    print(local_epochs, 'local_epochs')
    batch_size = 500
    EMNIST_powerfulCNN = args.powerfulCNN
    type_iid = args.iid_type  # 'iid', 'noniid'
    class_ratio = args.class_ratio  # non iid only
    beta = args.alpha  # default 3
    task_type = args.task_type
    task_number = len(task_type)
    data_ratio = args.data_ratio

    # venn diagram conditions setting
    # total task number
    venn_list = args.venn_list # [0.3,0.4,0.3]
    venn_matrix = np.zeros((task_number, num_clients))
    # 30% can handle , 40% can handle handle task_num-1, 30% can handle task_num-2
    #venn_task_num_list = [task_number, max(1,task_number-1), max(1,task_number-2)]
    venn_task_num_list = [1, 1, 1] # remove venn structure for the ease of implementation
    # each client can only train one model per round
    task_num_venn_list = []
    for i in range(num_clients):
        task_num_venn = random.choices(venn_task_num_list, venn_list, k=1)[0]
        task_num_venn_list.append(task_num_venn)
        # choose task_num_venn tasks, set them to 1 in venn_matrix
        # random decide the task indices
        task_indices = random.sample(range(task_number), task_num_venn)
        for idx in task_indices:
            venn_matrix[idx, i] = 1


    client_cpu_list = args.client_cpu
    assert sum(client_cpu_list) == 1.0
    # get task ability of each client
    # 3 levels: straggler, common, expert
    # straggler can only train 1 task, common can train half, expert can train all
    client_task_ability = []
    # random decide the ability of each client
    for i in range(num_clients):
        task_level_list = [1, max(1, task_num_venn_list[i] // 2), task_num_venn_list[i]]  # we simulate 3 levels of clients
        # make sure no get 0
        task_level_list = [max(1, x) for x in task_level_list]
        client_task_ability.append(random.choices(task_level_list, client_cpu_list)[0]) # randomly decide the ability of each client
    # expand total_clients by client_task_ability
    clients_process = []
    for i in range(num_clients):
        clients_process.extend([i] * client_task_ability[i])
    allowed_communication = int(C * len(clients_process))


    # set record name
    iid_str = ''.join([x[0] for x in type_iid])
    task_str = ''.join([x[0] for x in task_type])
    iid_filename = iid_str + task_str + '_a' + str(beta) + 'P' if EMNIST_powerfulCNN \
        else iid_str + task_str + '_a' + str(beta) + 'l'
    tasks_weight = np.ones(len(task_type)) / len(task_type)

    folder_name = str(task_number)+"task_"+iid_str+"_"+args.notes
    if not os.path.exists('./result/'+folder_name):
        os.makedirs('./result/'+folder_name)
    else:
        if args.insist is True:
            # go ahead
            print('folder exists but still go on!')
            # delete all files inside the folder
            import shutil
            shutil.rmtree('./result/'+folder_name)
            os.makedirs('./result/'+folder_name)

        else:
            print('folder exists!')
            sys.exit()

    print('task number', task_number)
    print('folder name', folder_name)
    print('exp num', exp_num)
    print('use powerful CNN?', EMNIST_powerfulCNN)
    print('iid type', type_iid)
    print('task type', task_type)

    for exp in range(0,exp_num):
        algorithm_name_vec = args.algo_type
        aggregation_dict = {'bayesian':'pkOverSumPk',
                            'proposed':'pkOverSumPk',
                            'random':'numUsersInv',
                            'alphafair': 'pkOverSumPk',
                            'round_robin':'numUsersInv'}

        aggregation_mtd_vec=[aggregation_dict[algo] for algo in algorithm_name_vec]

        for algo in range(len(algorithm_name_vec)):
            algorithm_name = algorithm_name_vec[algo]
            aggregation_mtd = aggregation_mtd_vec[algo]
            print('exp', exp, 'algo', algorithm_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#'cuda:0'

            #for round robin
            rr_taskAlloc=np.zeros(num_clients)
            first_ind=0
            for i in range(len(task_type)):
                rr_taskAlloc[first_ind:math.floor((i+1)*num_clients/len(task_type))]=i
                first_ind=math.floor((i+1)*num_clients/len(task_type))
            firstIndRR=0
            clients_task=0
            file =  open('./result/'+folder_name+'/Algorithm_'+algorithm_name+'_normalization_'+normalization+'_type_'+iid_filename+'_seed_'+str(random_seed)+''+'.txt', 'w')

            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(random_seed)

            global_models = []
            local_results = []
            global_results = []
            tasks_data_info = []
            tasks_data_idx = []
            globalAccResults=np.zeros((len(task_type), num_round))
            globalLossResults=np.zeros((len(task_type), num_round))
            localAccResults= np.zeros((len(task_type), num_clients, num_round))
            localLossResults= np.zeros((len(task_type), num_clients, num_round))
            allocation_dict_list = []
            old_local_updates = []
            old_local_updates_previous = []
            old_local_updates_epoch1 = []
            optimal_b_list = []
            decay_tasks_list = []
            decay_beta_record = np.zeros((num_round+1, len(task_type), num_clients))
            decay_rates = np.zeros((task_number, num_clients))

            TaskAllocCounter=np.zeros((len(task_type),num_round))

            for i in range(len(task_type)):
                if task_type[i] == 'shakespeare':
                    import utility.language_tools as language_tools
                    dataset_train = language_tools.ShakeSpeare(train=True)
                    dataset_test = language_tools.ShakeSpeare(train=False)
                    dict_users = dataset_train.get_client_dic()
                    # remove the key if the key is larger than num_clients
                    data_ratio = args.data_ratio - 0.1*i
                    dict_users = {key: dict_users[key] for key in range(num_clients)}
                    dict_users = [list(dict_users[key]) for key in dict_users] # become a list
                    dict_users = [d[:int(data_ratio*len(d))] for d in dict_users]
                    # only use part of the data
                    tasks_data_info.append([dataset_train, dataset_test, -1, -1, -1, -1])
                    # convert dict_users to tasks_data_idx
                    tasks_data_idx.append(dict_users)
                    global_models.append(load_model(name_data=task_type[i], num_classes=-1, args=args).to(device))
                else:
                    tasks_data_info.append(preprocessing(task_type[i], data_ratio, args)) # 0: trainset, 1: testset, 2: min_data_num, 3: max_data_num 4: input_size, 5: classes_size
                    if type_iid[i] =='iid':
                        tasks_data_idx.append(dataset.iid(dataset=tasks_data_info[i][0],
                                                        min_data_num=tasks_data_info[i][2],
                                                        max_data_num=tasks_data_info[i][3],
                                                        num_users=num_clients)) # 0: clients_data_idx
                    elif type_iid[i] =='noniid':
                        tasks_data_idx.append(dataset.noniid(dataset=tasks_data_info[i][0],
                                            min_data_num=tasks_data_info[i][2],
                                            max_data_num=tasks_data_info[i][3],
                                            class_ratio=class_ratio[i],
                                            num_users=num_clients)) # 0: clients_data_idx 1: clients_label
                    global_models.append(load_model(name_data=task_type[i], num_classes=tasks_data_info[i][5], args=args).to(device))
                local_results.append([0.1, 1])  # 0: acc, 1: loss
                global_results.append([0.1, 1])


            # record all client data num
            all_data_num = []
            for task_idx in range(len(task_type)):
                local_data_num = []
                for client_idx in range(num_clients):
                    if type_iid[task_idx] == 'iid' or task_type[task_idx] == 'shakespeare':
                        local_data_num.append(len(tasks_data_idx[task_idx][client_idx]))
                    elif type_iid[task_idx] == 'noniid':
                        local_data_num.append(len(tasks_data_idx[task_idx][0][client_idx]))
                all_data_num.append(local_data_num)

            # record data distribution, client labels
            for task_idx in range(len(task_type)):
                if type_iid[task_idx] == 'noniid':
                    for client_idx in range(num_clients):
                        distribution_file = './result/' + folder_name + '/' + f'client_labels_{task_idx}.pkl'
                        optimal_sampling.append_to_pickle(distribution_file, tasks_data_idx[task_idx][0][client_idx])
                    print('distribution saved')
            # compute dis
            all_data_array = np.array(all_data_num)
            # filter out the clients that can not handle the task
            all_data_array = all_data_array * venn_matrix
            # compute the sum of data num for each task
            all_data_sum = np.sum(all_data_array, axis=1)
            # compute data_num[i,s]/all_data_num[s]
            dis = np.zeros((len(task_type), num_clients))
            for task_idx in range(len(task_type)):
                for client_idx in range(num_clients):
                    dis[task_idx, client_idx] = all_data_array[task_idx, client_idx] / all_data_sum[task_idx]

            global_accs = []
            for task_idx in range(len(task_type)):
                global_accs.append(0.1)

            localLoss = np.zeros((task_number, num_clients))
            localLoss = get_local_loss(task_number, num_clients, task_type, type_iid, tasks_data_info,
                                       tasks_data_idx, global_models, device, batch_size, venn_matrix,
                                       False, localLoss, args.fresh_ratio)
            # initialize the localLoss matrix
            stored_wdiff_list = np.ones((task_number, num_clients)) * 0.5

            if args.scaffold is True:
                control_variate = [[] for _ in range(task_number)]
                for task_idx in range(len(task_type)):
                    for client_idx in range(num_clients):
                        control_variate[task_idx].append(optimal_sampling.zero_shapelike(global_models[task_idx].state_dict()))

            if (args.stale is True) or (args.approximation is True):
                old_local_updates = [[] for _ in range(task_number)]
                for task_idx in range(len(task_type)):
                    for client_idx in range(num_clients):
                        old_local_updates[task_idx].append(optimal_sampling.zero_shapelike(global_models[task_idx].state_dict()))


            old_local_updates_previous = [[] for _ in range(task_number)]
            for task_idx in range(len(task_type)):
                for client_idx in range(num_clients):
                    old_local_updates_previous[task_idx].append(
                        optimal_sampling.zero_shapelike(global_models[task_idx].state_dict()))


            # if lastLayer is True, similarly initialize old_local_updates_epoch1
            if args.lastLayer is True:
                old_local_updates_epoch1 = [[] for _ in range(task_number)]
                for task_idx in range(len(task_type)):
                    for client_idx in range(num_clients):
                        old_local_updates_epoch1[task_idx].append(optimal_sampling.zero_shapelike(global_models[task_idx].state_dict()))


            optimal_b_array = np.zeros((len(task_type), num_clients))
            adjusted_old_local_updates = copy.deepcopy(old_local_updates)
            recent_G = copy.deepcopy(old_local_updates)


            for round in tqdm(range(num_round)):
                print(f"Round[ {round+1}/{num_round} ]",file=file)
                # random sampling
                # all_clients = list(range(0, num_clients))
                #chosen_clients = random.sample(all_clients, int(numUsersSel))
                # random selection
                if args.fullparticipation is True:
                    chosen_clients = []
                    clients_task = []
                    for task in range(len(task_type)):
                        chosen_clients.extend(clients_process)
                        clients_task.extend([task] * len(clients_process))
                    localLoss = get_local_loss(task_number, num_clients, task_type, type_iid, tasks_data_info,
                                               tasks_data_idx, global_models, device, batch_size, venn_matrix,
                                               args.freshness, localLoss, args.fresh_ratio)
                    localLossResults[:, :, round] = localLoss
                else:
                    if args.givenProb != 0.0:
                        chosen_clients, clients_task, p_dict, p_all = optimal_sampling.sample_unbalanced_distribution(clients_process, int(allowed_communication), args.givenProb, task_number)
                        # include p_all to args.p_all
                        args.p_all = p_all
                    else:
                        chosen_clients = random.sample(clients_process, int(allowed_communication))
                        # allocate task based on venn matrix
                        clients_task = []
                        for process in chosen_clients:
                            clients_task.append(np.random.choice(np.where(venn_matrix[:, process] == 1)[0]))
                    # this will be used for random sampling
                # training
                if (args.optimal_sampling is True):
                    # train everything to get every gradient
                    # need to record local loss before the training
                    localLoss = get_local_loss(task_number, num_clients, task_type, type_iid, tasks_data_info,
                                               tasks_data_idx, global_models, device, batch_size, venn_matrix, args.freshness, localLoss, args.fresh_ratio)
                    localLossResults[:, :, round] = localLoss

                    if (args.approximation is True) and (round > 0):
                        # train randomly
                        tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, all_weights_diff = training(
                            tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                            global_models=global_models, chosen_clients=chosen_clients,
                            task_type=task_type, clients_task=clients_task,
                            local_epochs=local_epochs, batch_size=batch_size, classes_size=tasks_data_info,
                            type_iid=type_iid, device=device, args=args)

                        pseudo_all_tasks_gradients_list = copy.deepcopy(old_local_updates)
                        # update new ones
                        for i in range(len(chosen_clients)):
                            task = clients_task[i]
                            cl = chosen_clients[i]
                            pseudo_all_tasks_gradients_list[task][cl] = copy.deepcopy(tasks_gradients_list[i])
                            all_tasks_gradients_list = pseudo_all_tasks_gradients_list

                    else:
                        all_tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, \
                            all_weights_diff, all_tasks_weights_list_epoch1 = training_all(
                                                                                            tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                                                                                            global_models=global_models, chosen_clients=None,
                                                                                            task_type=task_type, clients_task=None,
                                                                                            local_epochs=local_epochs, batch_size=batch_size, classes_size=tasks_data_info,
                                                                                            type_iid=type_iid, device=device, args=args)

                      # normal methods, with closed-form solution, decay-approximation or optimal values.
                    if args.optimal_b is True:
                        if args.lastLayer is True:
                            optimal_b_array = optimal_sampling.get_optimal_b(all_tasks_weights_list_epoch1,
                                                                             old_local_updates_epoch1,
                                                                             task_number, num_clients, args)
                        else:
                            optimal_b_array = optimal_sampling.get_optimal_b(all_tasks_gradients_list, old_local_updates,
                                                                         task_number, num_clients, args)
                        optimal_b_list.append(optimal_b_array)
                    else:
                        optimal_b_array = decay_beta_record[round]
                    if args.stale is True:
                        for task_idx in range(len(task_type)):
                            for client_idx in range(num_clients):
                                adjusted_old_local_updates[task_idx][client_idx] = copy.deepcopy(
                                    optimal_sampling.newupdate(old_local_updates[task_idx][client_idx],
                                                               optimal_b_array[task_idx][client_idx]))

                    if args.stale is True:
                        if round == 0:
                            pseudo_all_weights_diff = copy.deepcopy(all_weights_diff)  # fast initialization
                            # old_local_updates = copy.deepcopy(all_tasks_gradients_list)  # fast initialization
                        else:
                            for task in range(task_number):
                                # the function is this-next.
                                # for us, new-old
                                # get learning rate for this task
                                # clients send norm(new-old) for sampling distribution
                                all_weights_diff = pseudo_all_weights_diff  # will update to other values, just an initialization to avoid all_weights_diff not yet defined
                                LR = optimizer_config(task_type[task])
                                for cl in range(num_clients):
                                    all_weights_diff[task][cl], _ = optimal_sampling.get_gradient_norm(
                                        weights_this_round=all_tasks_gradients_list[task][cl],
                                        weights_next_round=adjusted_old_local_updates[task][cl],
                                        lr=LR)

                    if args.freshness is True:
                        if round == 0:
                            stored_wdiff_list = all_weights_diff  # fast initialization
                        else:
                            if args.noextra_com is True:
                                # adjust old based on new beta
                                if args.adjustoldVR is True:  # GVR no need to adjust old
                                    # update recent_G
                                    if round == 0:
                                        recent_G = copy.deepcopy(all_tasks_gradients_list)
                                    else:
                                        for i in range(len(chosen_clients)):
                                            task = clients_task[i]
                                            client = chosen_clients[i]
                                            recent_G[task][client] = copy.deepcopy(all_tasks_gradients_list[task][client])
                                    # update based on recent_G
                                    for task in range(task_number):
                                        LR = optimizer_config(task_type[task])
                                        for client in range(num_clients):
                                            stored_wdiff_list[task][client], _ = optimal_sampling.get_gradient_norm(
                                                weights_this_round=recent_G[task][client],
                                                weights_next_round=adjusted_old_local_updates[task][client],
                                                lr=LR)
                                else:
                                    # adjust new updates as the latest
                                    for i in range(len(chosen_clients)):
                                        stored_wdiff_list[clients_task[i]][chosen_clients[i]] = all_weights_diff[clients_task[i]][chosen_clients[i]] \
                                                                                                * venn_matrix[clients_task[i], chosen_clients[i]]
                        all_weights_diff = stored_wdiff_list
                    # optimal sampling
                    all_weights_diff_power = all_weights_diff
                        #optimal_sampling.power_gradient_norm(all_weights_diff, localLoss, args, dis)
                    if args.skipOS is False:
                        clients_task, p_dict, chosen_clients = optimal_sampling.get_optimal_sampling(chosen_clients,
                                                                                         dis,
                                                                                         all_weights_diff_power, args, client_task_ability, clients_process, venn_matrix, save_path='./result/'+folder_name+'/')
                        # optimal sampling needs to be moved after we get local_data_nums



                        # if approximation, train again
                        if (args.approximation is True) and (round > 0):
                            tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, all_weights_diff = training(
                                tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                                global_models=global_models, chosen_clients=chosen_clients,
                                task_type=task_type, clients_task=clients_task,
                                local_epochs=local_epochs, batch_size=batch_size, classes_size=tasks_data_info,
                                type_iid=type_iid, device=device, args=args)

                            pseudo_all_tasks_gradients_list = copy.deepcopy(old_local_updates)
                            # update new ones
                            for i in range(len(chosen_clients)):
                                task = clients_task[i]
                                cl = chosen_clients[i]
                                pseudo_all_tasks_gradients_list[task][cl] = copy.deepcopy(tasks_gradients_list[i])
                                all_tasks_gradients_list = pseudo_all_tasks_gradients_list
                            # change to pseudo_all_tasks_gradients_list
                    else:
                        # if skipOS is True, then need to build p_dict given the chosen_clients and clients_task
                        if args.givenProb != 0.0:
                            pass
                        else:
                            p_dict = []
                            for task_index in range(task_number):
                                # track how many clients are chosen for this task: count how many task_index in clients_task
                                task_count = clients_task.count(task_index)
                                p_dict.append([args.C / task_number] * task_count)

                else:
                    # if args.approx_optimal, then get all local loss and acc, update chosen_clients and clients_task
                    if args.approx_optimal is True:
                        # 1 get all local loss
                        localLoss = get_local_loss(task_number, num_clients, task_type, type_iid, tasks_data_info,
                                                   tasks_data_idx, global_models, device, batch_size, venn_matrix, args.freshness, localLoss, args.fresh_ratio)
                        if args.acc is True:
                            localLoss = get_local_acc(task_number, num_clients, task_type, type_iid, tasks_data_info,
                                                   tasks_data_idx, global_models, device, batch_size, venn_matrix, args.freshness, localLoss, args.fresh_ratio)
                        localLossResults[:, :, round] = localLoss
                        # use loss to replace gradient norm
                        all_weights_diff_power = optimal_sampling.power_gradient_norm(localLoss,
                                                                              localLoss, args,
                                                                              dis)
                        clients_task, p_dict, chosen_clients = optimal_sampling.get_optimal_sampling(chosen_clients,
                                                                                                 dis,
                                                                                                 all_weights_diff_power, args, client_task_ability,
                                                                                                clients_process, venn_matrix, save_path='./result/'+folder_name+'/')

                    if args.scaffold is True:
                        tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, all_weights_diff, control_variate = training_scaffold(
                            tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                            global_models=global_models, chosen_clients=chosen_clients,
                            task_type=task_type, clients_task=clients_task,
                            local_epochs=local_epochs, batch_size=batch_size, classes_size=tasks_data_info,
                            type_iid=type_iid, device=device, args=args, control_variate=control_variate, dis=dis)
                    else:
                        tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, all_weights_diff = training(tasks_data_info=tasks_data_info, tasks_data_idx=tasks_data_idx,
                                                                                                   global_models=global_models, chosen_clients=chosen_clients,
                                                                                                   task_type=task_type, clients_task=clients_task,
                                                                                                   local_epochs=local_epochs, batch_size = batch_size, classes_size = tasks_data_info,
                                                                                                    type_iid=type_iid, device=device, args=args)
                        if (args.stale is True) or (args.approximation is True):
                            pseudo_all_tasks_gradients_list = copy.deepcopy(old_local_updates)
                            # update new ones
                            for i in range(len(chosen_clients)):
                                task = clients_task[i]
                                cl = chosen_clients[i]
                                pseudo_all_tasks_gradients_list[task][cl] = copy.deepcopy(tasks_gradients_list[i])
                                all_tasks_gradients_list = pseudo_all_tasks_gradients_list


                allocation_dict = {}
                for i in range(len(chosen_clients)):
                    # if clients_task[i] is not a key in allocation_dict, then add it
                    if chosen_clients[i] in allocation_dict:
                        allocation_dict[chosen_clients[i]].append(clients_task[i])
                    else:
                        allocation_dict[chosen_clients[i]] = [clients_task[i]]
                allocation_dict_list.append(allocation_dict)



                if args.optimal_sampling is True:
                    # remember to process local_loss
                    temp_global_results = []
                    for task_idx in range(len(task_type)):
                        this_task_gradients_list = []
                        this_task_chosen_clients = []
                        # get local_weights for this task
                        for client_idx in range(len(chosen_clients)):
                            if clients_task[client_idx] == task_idx:
                                this_task_gradients_list.append(all_tasks_gradients_list[task_idx][chosen_clients[client_idx]])
                                this_task_chosen_clients.append(chosen_clients[client_idx])
                        assert len(this_task_gradients_list) == len(p_dict[task_idx])
                        # aggregation
                        LR = optimizer_config(task_type[task_idx])

                        if (len(this_task_gradients_list) != 0):
                            if args.cpumodel is True:
                                global_models[task_idx].to('cpu')

                            if args.stale is True:
                                global_models[task_idx].load_state_dict(
                                    federated_stale(global_weights=global_models[task_idx],
                                                   models_gradient_dict=this_task_gradients_list,
                                                   local_data_num=dis[task_idx],
                                                   p_list=p_dict[task_idx], args=args, decay_beta=decay_beta_record[round, task_idx], chosen_clients=this_task_chosen_clients,
                                                   old_global_weights=adjusted_old_local_updates[task_idx], old_global_weights_previous=old_local_updates_previous[task_idx], allocation_result=allocation_dict_list, task_index=task_idx,
                                                    save_path='./result/'+folder_name+'/', allnew_gradients=all_tasks_gradients_list[task_idx]))
                            else:
                                global_models[task_idx].load_state_dict(
                                federated_prob(global_weights=global_models[task_idx], models_gradient_dict=this_task_gradients_list, local_data_num=dis[task_idx],
                                          p_list=p_dict[task_idx], args=args, chosen_clients=chosen_clients, tasks_local_training_loss=localLoss[task_idx], lr=LR))
                            if args.cpumodel is True:
                                global_models[task_idx].to(device)
                            temp_global_results.append(
                                evaluation(model=global_models[task_idx], data=tasks_data_info[task_idx][1],
                                           batch_size=batch_size, device=device, args=args))
                            # print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                            print('p_list', p_dict[task_idx], file=file)
                            #print('p_list', p_dict[task_idx])
                            print(
                                f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}",
                                file=file)
                            #print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                        else:
                            temp_global_results.append(global_results[task_idx])
                            # print(f"Task[{task_idx}]: Global not changed")
                            print(f"Task[{task_idx}]: Global not changed", file=file)

                    global_accs = []
                    for task_idx in range(len(task_type)):
                        global_accs.append(temp_global_results[task_idx][0])
                else:
                    temp_global_results = []
                    if (algorithm_name == 'random') or (algorithm_name == "round_robin") or (algorithm_name == "alphafair"):
                        localLoss = np.zeros((task_number, num_clients))
                    for task_idx in range(len(task_type)):
                        temp_local_gradients = []
                        temp_local_P = []
                        this_task_chosen_clients = []

                        for clients_idx, local_gradients in enumerate(tasks_gradients_list):
                            if clients_task[clients_idx] == task_idx:
                                temp_local_gradients.append(local_gradients)
                                this_task_chosen_clients.append(chosen_clients[clients_idx])
                                if args.approx_optimal is True:
                                    temp_local_P = p_dict[task_idx]
                                elif args.approximation is True:
                                    if round == 0:
                                        temp_local_P.append(C/task_number)
                                    else:
                                        temp_local_P = p_dict[task_idx]  # define the same for many times, but it is ok here
                                else:
                                    if args.fullparticipation is True:
                                        p = 1.0
                                        temp_local_P.append(p)
                                    else:
                                        p = C/task_number
                                        temp_local_P.append(p)
                                # do not need to collect local_data num, just use all_local_data_num
                        # aggregation
                        LR = optimizer_config(task_type[task_idx])
                        if (len(temp_local_gradients) != 0):
                            if args.cpumodel is True:
                                global_models[task_idx].to('cpu')
                            if args.stale is True:
                                global_models[task_idx].load_state_dict(
                                    federated_stale(global_weights=global_models[task_idx],
                                                    models_gradient_dict=temp_local_gradients,
                                                    local_data_num=dis[task_idx],
                                                    p_list=temp_local_P, args=args,
                                                    decay_beta=decay_beta_record[round, task_idx],
                                                    chosen_clients=this_task_chosen_clients,
                                                    old_global_weights=old_local_updates[task_idx],
                                                    allocation_result=allocation_dict_list, task_index=task_idx,
                                                    save_path='./result/' + folder_name + '/'))
                            else:
                                global_models[task_idx].load_state_dict(
                                federated_prob(global_weights=global_models[task_idx],
                                               models_gradient_dict=temp_local_gradients,
                                               local_data_num=dis[task_idx],
                                               p_list=temp_local_P, args=args, chosen_clients=this_task_chosen_clients, tasks_local_training_loss=localLoss[task_idx], lr=LR))
                            print('p_list', temp_local_P, file=file)
                            if args.cpumodel is True:
                                global_models[task_idx].to(device)
                            temp_global_results.append(evaluation(model = global_models[task_idx], data = tasks_data_info[task_idx][1], batch_size = batch_size, device = device, args=args))
                            print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}",file=file)
                            #print(f"Task[{task_idx}]: Global Acc-{temp_global_results[task_idx][0]} Global Loss-{temp_global_results[task_idx][1]}")
                        else:
                            temp_global_results.append(global_results[task_idx])
                            #print(f"Task[{task_idx}]: Global not changed")
                            print(f"Task[{task_idx}]: Global not changed",file=file)

                    global_accs = []
                    for task_idx in range(len(task_type)):
                        global_accs.append(temp_global_results[task_idx][0])
                # update stale after aggregation
                # define pseudo_all_tasks_gradients_list for format matching

                if (args.stale is True) and (args.optimal_sampling is True):
                    # only update old_local_updates for chosen_clients and tasks
                    if (args.optimal_b is False) and (args.stale_b0 == 0):  # dynamic b0
                        # normal way, to approximate optimal lambda
                        b0 = 0.9
                        decay_rates = optimal_sampling.approximate_decayb(new_updates=all_tasks_gradients_list, old_updates=old_local_updates,
                                                           tasknum=task_number, clientnum=num_clients,
                                                           allocation_record=allocation_dict_list, chosen_clients=chosen_clients,
                                                           client_task=clients_task, b0=b0, decay_rates=decay_rates)
                        decay_tasks_list.append(decay_rates)
                        # let everything decay in the next round
                        for task in range(task_number):  # record decayed beta
                            for client in range(num_clients):
                                decay_beta_record[round + 1, task, client] = max(0.0, decay_beta_record[round, task, client] - decay_rates[task, client])
                        # initialize new updates
                        for i in range(len(chosen_clients)):
                            task = clients_task[i]
                            client = chosen_clients[i]
                            # recover to the original scale
                            # if decay_beta_record[round + 1, task, client] == 0, first time active, scale to b0 directly
                            decay_beta_record[round + 1, task, client] = b0
                    else:  # fixed value of beta, no decay (FedStale)
                        b0 = args.stale_b0
                        b_tasks = [args.stale_b for _ in range(task_number)]
                        decay_tasks_list.append(b_tasks)

                        for task in range(task_number):  # record decayed beta
                            decay_beta_record[round + 1, task, :] = decay_beta_record[round, task, :]
                        # initialize new updates
                        for i in range(len(chosen_clients)):
                            task = clients_task[i]
                            client = chosen_clients[i]
                            # recover to the original scale
                            # if decay_beta_record[round + 1, task, client] == 0, first time active, scale to b0 directly
                            decay_beta_record[round + 1, task, client] = b0

                if args.stale is True:
                    # update old_local_updates
                    for i in range(len(chosen_clients)):
                        task = clients_task[i]
                        cl = chosen_clients[i]

                        if args.effV is True:
                            old_local_updates_previous[task][cl] = copy.deepcopy(old_local_updates[task][cl])
                            old_local_updates[task][cl] = copy.deepcopy(all_tasks_gradients_list[task][cl])
                        else:
                            old_local_updates[task][cl] = copy.deepcopy(all_tasks_gradients_list[task][cl])
                        if args.lastLayer is True:
                            old_local_updates_epoch1[task][cl] = copy.deepcopy(all_tasks_weights_list_epoch1[task][cl])


                TaskAllocCounter[:, round] = np.bincount(np.array(clients_task).astype(np.int64), minlength=len(task_type))
                #print("alloc", TaskAllocCounter[:, round])

                #local_results = temp_local_results
                global_results = temp_global_results

                globalAccResults[:,round]=np.array(temp_global_results)[:,0]
                #localAccResults[:,round]=np.array(temp_local_results)[:,0]
                globalLossResults[:,round]=np.array(temp_global_results)[:,1]
                #localLossResults[:,round]=np.array(temp_local_results)[:,1]

                # record global model performance on local client data
            # only record the last round performance
            localAcc = np.zeros((task_number, num_clients))
            localLoss = np.zeros((task_number, num_clients))
            for cl in range(num_clients):
                for task in range(task_number):
                    if type_iid[task] == 'iid' or task_type[task] == 'shakespeare':
                        client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][
                            cl])  # or iid_partition depending on your choice
                    elif type_iid[task] == 'noniid':
                        client_data = Subset(tasks_data_info[task][0], tasks_data_idx[task][0][
                            cl])  # or iid_partition depending on your choice
                    accu, loss = evaluation(model=global_models[task], data=client_data,
                                            batch_size=batch_size, device=device, args=None)  # use all data
                    #localAcc[task, cl] = accu
                    #localLoss[task, cl] = loss
                    localAccResults[task, cl, -1] = accu
                    localLossResults[task, cl, -1] = loss

            # group fairness evaluation
            for task_idx in range(task_number):
                results = group_fairness_evaluation(model=global_models[task_idx], data=tasks_data_info[task_idx][1],
                                          batch_size=batch_size, device=device, args=args)
                results_file = './result/'+folder_name+'/'+f'groupfairness_results_task{task_idx}.pkl'
                with open(results_file, 'wb') as f:
                    pickle.dump(results, f)




            filename='mcf_i_globalAcc_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, globalAccResults)
            filename='mcf_i_globalLoss_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, globalLossResults)

            filename='localAcc_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, localAccResults)
            filename='localLoss_exp{}_algo{}.npy'.format(exp,algo)
            np.save('./result/'+folder_name+'/'+ filename, localLossResults)

            filename = 'allocCounter_{}.npy'.format(algo)
            np.save('./result/'+folder_name+'/'+filename, TaskAllocCounter)

            # save allocation_dict_list
            # allocation_dict_list is a list, each element is a dict
            # each dict is the allocation of one round
            import pickle
            filename = 'allocation_dict_list_{}.pkl'.format(algo)
            with open('./result/'+folder_name+'/'+filename, 'wb') as f:
                pickle.dump(allocation_dict_list, f)

            # store optimal_b_list
            filename = 'optimal_b_list.pkl'
            with open('./result/'+folder_name+'/'+filename, 'wb') as f:
                pickle.dump(optimal_b_list, f)

            filename = 'decay_beta_record.pkl'
            with open('./result/'+folder_name+'/'+filename, 'wb') as f:
                pickle.dump(decay_beta_record, f)

            # save venn_matrix
            filename = 'venn_matrix.pkl'
            with open('./result/'+folder_name+'/'+filename, 'wb') as f:
                pickle.dump(venn_matrix, f)
            # save client_task_ability
            filename = 'client_task_ability.pkl'
            with open('./result/'+folder_name+'/'+filename, 'wb') as f:
                pickle.dump(client_task_ability, f)
            # save dis
            filename = 'dis.pkl'
            with open('./result/'+folder_name+'/'+filename, 'wb') as f:
                pickle.dump(dis, f)

            # save decay_tasks_list
            filename = 'decay_tasks_list.pkl'
            with open('./result/' + folder_name + '/' + filename, 'wb') as f:
                pickle.dump(decay_tasks_list, f)

            print(f'Finished Training, lr:{args.lr}, Global acc:{globalAccResults[:, -1]}')
            print(f'Finished Training, lr:{args.lr}, Global acc:{globalAccResults[:, -1]}', file=file)
            file.close()