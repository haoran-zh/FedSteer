import torch
from torch.utils.data import Subset, DataLoader
from utility.load_model import load_model
from utility.config import optimizer_config
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import utility.optimal_sampling as optimal_sampling
import numpy as np
from utility.language_tools import DatasetSplit
import torch.nn.functional as F
from utility.scaffold import ScaffoldOptimizer
from utility.optimal_sampling import zero_shapelike


def state_dict_to_gpu(state_dict, device):
    # Create a new dictionary to hold the GPU tensors
    gpu_state_dict = {}

    for key, value in state_dict.items():
        # Move each tensor to the GPU
        gpu_state_dict[key] = value.to(device)

    return gpu_state_dict

def state_dict_to_cpu(state_dict):
    # Create a new dictionary to hold the GPU tensors
    cpu_state_dict = {}

    for key, value in state_dict.items():
        # Move each tensor to the GPU
        cpu_state_dict[key] = value.cpu()

    return cpu_state_dict


def compute_adjusted_gradients(lr_gradients, learning_rate, local_epochs_task_idx):
    # Create a new dictionary to hold the adjusted gradients
    adjusted_gradients = {}

    # Iterate over each key-value pair in lr_gradients
    for key, value in lr_gradients.items():
        # Perform the required computation
        adjusted_value = value.clone() / learning_rate / local_epochs_task_idx * (-1)
        # Add the adjusted value to the new dictionary
        adjusted_gradients[key] = adjusted_value

    return adjusted_gradients


class AlphaFairnessLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(AlphaFairnessLoss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, labels, client_labels=None):
        # outputs: predictions from the model
        # labels: true labels
        # client_labels: list of unique labels for the client (optional)

        if client_labels is None:
            client_labels = torch.unique(labels)

        alpha_loss = 0.0
        for label in client_labels:
            mask = (labels == label)
            if mask.sum() == 0:
                continue
            label_outputs = outputs[mask]
            label_labels = labels[mask]
            ce_loss = F.cross_entropy(label_outputs, label_labels, reduction='sum')
            alpha_loss += ce_loss.pow(self.alpha)
        # compute the mean
        # number of samples
        num_samples = outputs.size(0)
        alpha_loss = alpha_loss / num_samples

        return alpha_loss

def training(tasks_data_info, tasks_data_idx, global_models, chosen_clients, task_type, clients_task, local_epochs,
             batch_size, classes_size, type_iid, device, args):
    # chosen_clients=[3,10,15]
    # clients_task=[0, 0, 0]
    weights_diff_list = []
    tasks_local_training_acc = []
    tasks_local_training_loss = []
    tasks_gradients_list = []
    # find out which task index is shakespeare
    tasks_list = args.task_type
    # find if we have shakespeare in the task list, if yes, get the index, if no, return -1
    shakespeare_index = [i for i, x in enumerate(tasks_list) if x == "shakespeare"]

    # print(clients_task, 'cl task')
    for data_idx, task_idx in enumerate(clients_task):
        task_idx = int(task_idx)
        local_model = load_model(name_data=task_type[int(task_idx)], num_classes=classes_size[task_idx][5],
                                 args=args).to(device)

        global_model_state_dict = global_models[int(task_idx)].state_dict()
        local_model_state_dict = local_model.state_dict()

        for key in local_model_state_dict.keys():
            local_model_state_dict[key] = global_model_state_dict[key]
        local_model.load_state_dict(local_model_state_dict)

        previous_local_state_dict = local_model_state_dict.copy()

        # Create a local optimizer
        learning_rate = optimizer_config(task_type[int(task_idx)])
        if task_idx in shakespeare_index:
            learning_rate = 1.4
        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
        if args.mse is True:
            local_criterion = nn.MSELoss()
        else:
            local_criterion = nn.CrossEntropyLoss()

        # Get client's data
        if type_iid[task_idx] == 'iid' or task_idx in shakespeare_index:
            client_data = Subset(tasks_data_info[task_idx][0], tasks_data_idx[task_idx][
                chosen_clients[data_idx]])  # or iid_partition depending on your choice
        elif type_iid[task_idx] == 'noniid':
            client_data = Subset(tasks_data_info[task_idx][0], tasks_data_idx[task_idx][0][
                chosen_clients[data_idx]])  # or iid_partition depending on your choice
            client_label = tasks_data_idx[task_idx][1][chosen_clients[data_idx]]
        client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
        local_model.train()
        local_train_accuracy = 0

        for _ in range(local_epochs[task_idx]):
            correct = 0
            running_loss = 0
            total = 0
            for images, labels in client_loader:
                images = images.to(device)
                labels = labels.to(device)
                local_optimizer.zero_grad()

                outputs = local_model(images)
                if type_iid[task_idx] == 'noniid' and task_idx not in shakespeare_index:
                    label_mask = torch.zeros(classes_size[task_idx][5], device=outputs.device)
                    label_mask[client_label] = 1
                    outputs = outputs.masked_fill(label_mask == 0, 0)
                loss = local_criterion(outputs, labels)
                loss.backward()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                local_optimizer.step()
            local_train_accuracy = correct / total
            local_train_loss = running_loss / total

        tasks_local_training_acc.append(local_train_accuracy)
        tasks_local_training_loss.append(local_train_loss)
        norm, lr_gradients = optimal_sampling.get_gradient_norm(previous_local_state_dict, local_model.state_dict(),
                                                                learning_rate)
        # Append local model weights to list
        if args.cpumodel is True:
            local_model.to('cpu')
        tasks_gradients_list.append(lr_gradients.copy())
        weights_diff_list.append(norm)

        # take a step once every global epoch
        # scheduler.step()

    return tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, weights_diff_list


def training_all(tasks_data_info, tasks_data_idx, global_models, chosen_clients, task_type, clients_task, local_epochs,
             batch_size, classes_size, type_iid, device, args):
    # args.lastLayer, use this to store the last layer gradient from the first epoch
    lastLayer = args.lastLayer  # if lastLayer is True, need to store the last layer gradient from the first epoch
    all_tasks_weights_list_epoch1 = []
    all_tasks_weights_list = []
    all_tasks_local_training_acc = []
    all_tasks_local_training_loss = []
    all_weights_diff = []
    tasks_list = args.task_type
    shakespeare_index = [i for i, x in enumerate(tasks_list) if x == "shakespeare"]
    # we want to get a list of weights, weights_list[task_index][client_index].
    # client_index is in the order of chosen_clients

    for tasks_index in range(len(task_type)): # all tasks and all clients need to be trained
        tasks_weights_list = []
        tasks_weights_list_epoch1 = []
        tasks_local_training_acc = []
        tasks_local_training_loss = []
        weights_diff = []
        for client_index in range(args.num_clients):
            local_model = load_model(name_data=task_type[int(tasks_index)], num_classes=classes_size[tasks_index][5],
                                     args=args).to(device)

            global_model_state_dict = global_models[int(tasks_index)].state_dict()
            local_model_state_dict = local_model.state_dict()

            for key in local_model_state_dict.keys():
                local_model_state_dict[key] = global_model_state_dict[key]
            local_model.load_state_dict(local_model_state_dict)

            previous_local_state_dict = local_model_state_dict.copy()

            # Create a local optimizer
            learning_rate = optimizer_config(task_type[tasks_index])
            # learning_rate = args.lr
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            # learning rate scheduler
            # scheduler = lr_scheduler.StepLR(local_optimizer, step_size=lr_step_size, gamma=gamma)
            # scheduler = lr_scheduler.MultiStepLR(local_optimizer, milestones=milestones, gamma=gamma)
            if args.mse is True:
                local_criterion = nn.MSELoss()
            else:
                local_criterion = nn.CrossEntropyLoss()

                # Get client's data
            if type_iid[tasks_index] == 'iid' or tasks_index in shakespeare_index:
                client_data = Subset(tasks_data_info[tasks_index][0], tasks_data_idx[tasks_index][client_index])  # or iid_partition depending on your choice
            elif type_iid[tasks_index] == 'noniid':
                client_data = Subset(tasks_data_info[tasks_index][0], tasks_data_idx[tasks_index][0][client_index])  # or iid_partition depending on your choice
                client_label = tasks_data_idx[tasks_index][1][client_index]
            client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
            local_model.train()
            local_train_accuracy = 0

            for epoch in range(local_epochs[tasks_index]):
                correct = 0
                running_loss = 0
                total = 0
                for images, labels in client_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    local_optimizer.zero_grad()
                    outputs = local_model(images)
                    if type_iid[tasks_index] == 'noniid' and tasks_index not in shakespeare_index:
                        label_mask = torch.zeros(classes_size[tasks_index][5], device=outputs.device)
                        label_mask[client_label] = 1
                        outputs = outputs.masked_fill(label_mask == 0, 0)
                    # print(outputs.shape)
                    # print(labels.shape)
                    loss = local_criterion(outputs, labels)
                    loss.backward()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    running_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    local_optimizer.step()
                local_train_accuracy = correct / total
                local_train_loss = running_loss / total

                if lastLayer is True and epoch == 0:
                    norm, lr_gradients = optimal_sampling.get_gradient_norm(previous_local_state_dict,
                                                                            local_model.state_dict(), learning_rate)
                    tasks_weights_list_epoch1.append(lr_gradients.copy())

            tasks_local_training_acc.append(local_train_accuracy) # only save the last epoch's accuracy
            tasks_local_training_loss.append(local_train_loss)
            # Append local model weights to list
            norm, lr_gradients = optimal_sampling.get_gradient_norm(previous_local_state_dict, local_model.state_dict(), learning_rate)
            # new lr = original_lr * sum(f^alpha-1(w_t))
            weights_diff.append(norm) # the norm without learning rate

            if args.cpumodel is True:
                local_model.to('cpu')
            tasks_weights_list.append(lr_gradients.copy()) # lr_gradient considers the learning rate
            # take a step once every global epoch
            # scheduler.step()
        all_tasks_weights_list_epoch1.append(tasks_weights_list_epoch1)
        all_tasks_weights_list.append(tasks_weights_list)
        all_tasks_local_training_acc.append(tasks_local_training_acc)
        all_tasks_local_training_loss.append(tasks_local_training_loss)
        all_weights_diff.append(weights_diff)
        # all_xxx list is in shape: [task_index][client_index]
        # client_index is in the order of chosen_clients[0], chosen_clients[1], chosen_clients[2]...
    return all_tasks_weights_list, all_tasks_local_training_acc, all_tasks_local_training_loss, all_weights_diff, all_tasks_weights_list_epoch1  # weights_diff is actually norm


def get_server_controls(control_variate, dis):
    server_control_list = []
    for task_idx in range(len(control_variate)):
        server_controls = zero_shapelike(control_variate[task_idx][0])
        global_keys = list(server_controls.keys())
        for client_idx in range(len(control_variate[0])):
            for key in global_keys:
                server_controls[key] += control_variate[task_idx][client_idx][key] * dis[task_idx][client_idx]
        server_control_list.append(server_controls)
    return server_control_list



def training_scaffold(tasks_data_info, tasks_data_idx, global_models, chosen_clients, task_type, clients_task, local_epochs,
             batch_size, classes_size, type_iid, device, args,
                      control_variate, dis):
    # chosen_clients=[3,10,15]
    # clients_task=[0, 0, 0]
    # get server control_variate
    server_control_list = get_server_controls(control_variate, dis)




    weights_diff_list = []
    tasks_local_training_acc = []
    tasks_local_training_loss = []
    tasks_gradients_list = []
    # find out which task index is shakespeare
    tasks_list = args.task_type
    # find if we have shakespeare in the task list, if yes, get the index, if no, return -1
    shakespeare_index = [i for i, x in enumerate(tasks_list) if x == "shakespeare"]

    # print(clients_task, 'cl task')
    for data_idx, task_idx in enumerate(clients_task):
        task_idx = int(task_idx)
        local_model = load_model(name_data=task_type[int(task_idx)], num_classes=classes_size[task_idx][5],
                                 args=args).to(device)

        global_model_state_dict = global_models[int(task_idx)].state_dict()
        local_model_state_dict = local_model.state_dict()

        for key in local_model_state_dict.keys():
            local_model_state_dict[key] = global_model_state_dict[key]
        local_model.load_state_dict(local_model_state_dict)

        previous_local_state_dict = local_model_state_dict.copy()

        # Create a local optimizer
        learning_rate = optimizer_config(task_type[int(task_idx)])
        if task_idx in shakespeare_index:
            learning_rate = 1.4
        local_optimizer = ScaffoldOptimizer(local_model.parameters(), lr=learning_rate, weight_decay=0.0)
        if args.mse is True:
            local_criterion = nn.MSELoss()
        else:
            local_criterion = nn.CrossEntropyLoss()

        # Get client's data
        if type_iid[task_idx] == 'iid' or task_idx in shakespeare_index:
            client_data = Subset(tasks_data_info[task_idx][0], tasks_data_idx[task_idx][
                chosen_clients[data_idx]])  # or iid_partition depending on your choice
        elif type_iid[task_idx] == 'noniid':
            client_data = Subset(tasks_data_info[task_idx][0], tasks_data_idx[task_idx][0][
                chosen_clients[data_idx]])  # or iid_partition depending on your choice
            client_label = tasks_data_idx[task_idx][1][chosen_clients[data_idx]]
        client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
        local_model.train()
        local_train_accuracy = 0
        # to gpu
        control_variate[task_idx][chosen_clients[data_idx]] = state_dict_to_gpu(control_variate[task_idx][chosen_clients[data_idx]], device)
        server_control_list[task_idx] = state_dict_to_gpu(server_control_list[task_idx], device)
        for _ in range(local_epochs[task_idx]):
            correct = 0
            running_loss = 0
            total = 0

            for images, labels in client_loader:
                images = images.to(device)
                labels = labels.to(device)
                local_optimizer.zero_grad()

                outputs = local_model(images)
                if type_iid[task_idx] == 'noniid' and task_idx not in shakespeare_index:
                    label_mask = torch.zeros(classes_size[task_idx][5], device=outputs.device)
                    label_mask[client_label] = 1
                    outputs = outputs.masked_fill(label_mask == 0, 0)
                loss = local_criterion(outputs, labels)
                loss.backward()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                # put server_control_list[task_idx] to to(device) gpu


                local_optimizer.step(server_controls=server_control_list[task_idx], client_controls=control_variate[task_idx][chosen_clients[data_idx]])
            local_train_accuracy = correct / total
            local_train_loss = running_loss / total

        tasks_local_training_acc.append(local_train_accuracy)
        tasks_local_training_loss.append(local_train_loss)
        norm, lr_gradients = optimal_sampling.get_gradient_norm(previous_local_state_dict, local_model.state_dict(),
                                                                learning_rate)
        # Append local model weights to list
        if args.cpumodel is True:
            local_model.to('cpu')
        tasks_gradients_list.append(lr_gradients.copy())
        weights_diff_list.append(norm)

        # take a step once every global epoch
        # scheduler.step()
        # update control variate
        control_variate[task_idx][chosen_clients[data_idx]] = state_dict_to_cpu(
            control_variate[task_idx][chosen_clients[data_idx]])
        server_control_list[task_idx] = state_dict_to_cpu(server_control_list[task_idx])

        temp = compute_adjusted_gradients(lr_gradients.copy(), learning_rate, local_epochs[task_idx])
        control_variate[task_idx][chosen_clients[data_idx]] = optimal_sampling.weight_minus(control_variate[task_idx][chosen_clients[data_idx]],
                                                                                            server_control_list[task_idx])
        control_variate[task_idx][chosen_clients[data_idx]] = optimal_sampling.weight_minus(control_variate[task_idx][chosen_clients[data_idx]],
                                                                                            temp)



    return tasks_gradients_list, tasks_local_training_acc, tasks_local_training_loss, weights_diff_list, control_variate