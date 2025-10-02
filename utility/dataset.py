import torch
import numpy as np
import sys

def iid(dataset, min_data_num, max_data_num, num_users):
    labels = torch.tensor(dataset.targets)
    num_classes = len(labels.unique())
    classes_list = []
    for c in range(num_classes):
        class_list = []
        for idx, label in enumerate(labels):
            if label==c:
                class_list.append(idx)
        classes_list.append(class_list)


    clients_data_idx = []
    classes_data_idx = np.zeros(num_classes, dtype=int)
    for i in range(num_users):
        random_number = np.random.randint(min_data_num[i], max_data_num[i]+1)

        uniform_idx=np.random.randint(0, num_classes)
        client_data_idx = []
        loop_counter = 0
        for _ in range(random_number):
            while(classes_data_idx[uniform_idx] >= len(classes_list[uniform_idx])):
                # len(classes_list[uniform_idx]) how many data points in class=uniform_idx
                # classes_data_idx[uniform_idx]  how many data points in class=uniform_idx have been assigned to clients
                # if all data points in class=uniform_idx have been assigned to clients, then uniform_idx+=1 to jump over this class
                uniform_idx += 1
                if uniform_idx == num_classes:
                    uniform_idx = 0
                loop_counter += 1
                if loop_counter > 3*num_classes:
                    print("data not enough")
                    sys.exit()

            client_data_idx.append(classes_list[uniform_idx][classes_data_idx[uniform_idx]])
            classes_data_idx[uniform_idx] += 1
            uniform_idx += 1
            if uniform_idx == num_classes:
                uniform_idx = 0
        clients_data_idx.append(client_data_idx)
    return clients_data_idx

# def noniid(dataset, min_data_num, max_data_num, class_ratio, num_users):
#     labels = torch.tensor(dataset.targets)
#     num_classes = len(labels.unique())
#     classes_list = []
#     for c in range(num_classes):
#         class_list = []
#         for idx, label in enumerate(labels):
#             if label==c:
#                 class_list.append(idx)
#         classes_list.append(class_list)
#
#
#     classes_len_list = []
#     for class_list in classes_list:
#         classes_len_list.append(len(class_list))
#
#     noniid_class_num = int(class_ratio*num_classes)
#
#     clients_data_idx = []
#     classes_data_idx = np.zeros(num_classes, dtype=int)
#     clients_label =[]
#     for i in range(num_users):
#         random_number = np.random.randint(min_data_num[i], max_data_num[i] + 1)
#         noniid_labels = []
#         # Clone classes_len_list so as not to mutate the original
#         temp_classes_len_list = classes_len_list.copy()
#         for _ in range(noniid_class_num):
#             noniid_label = np.random.choice(np.where(temp_classes_len_list==np.max(temp_classes_len_list))[0],1)[0]
#             temp_classes_len_list[noniid_label] = -1
#             noniid_labels.append(noniid_label)
#         clients_label.append(noniid_labels)
#         uniform_idx = np.random.randint(0, noniid_class_num)
#         client_data_idx = []
#         for _ in range(random_number):
#             #print(uniform_idx)
#             #print(len(noniid_labels))
#             #print(classes_data_idx[noniid_labels[uniform_idx]])
#             #print(len(classes_list[noniid_labels[uniform_idx]]))
#             while(classes_data_idx[noniid_labels[uniform_idx]] > len(classes_list[noniid_labels[uniform_idx]])):
#                 uniform_idx += 1
#                 if uniform_idx == noniid_class_num:
#                     uniform_idx = 0
#             client_data_idx.append(classes_list[noniid_labels[uniform_idx]][classes_data_idx[noniid_labels[uniform_idx]]])
#             classes_data_idx[noniid_labels[uniform_idx]] += 1
#             classes_len_list[noniid_labels[uniform_idx]] -= 1
#             uniform_idx += 1
#             if uniform_idx == noniid_class_num:
#                 uniform_idx = 0
#         clients_data_idx.append(client_data_idx)
#     return clients_data_idx, clients_label


def noniid(dataset, min_data_num, max_data_num, num_users, class_ratio):
    """
    Creates a non-IID distribution combining a bipartite structure with k-class client selection.

    1. Total samples are determined by SUM(AVG(min_data_num, max_data_num)).
    2. Clients are split into majority/minority groups with access to common/rare class pools.
    3. For each client, a subset of k classes is chosen from its pool, where k is
       controlled by class_ratio.
    4. The sample budget is then distributed proportionally among clients according to these rules.

    Args:
        dataset: The entire dataset.
        min_data_num (list): List of min relative sizes for each client.
        max_data_num (list): List of max relative sizes for each client.
        num_users (int): The total number of clients.
        class_ratio (float): The ratio of classes to assign to each client from their pool.

    Returns:
        tuple: A tuple containing:
            - clients_data_idx (list of lists): The data indices for each client.
            - clients_label_map (dict): A dictionary mapping each client_id to its final assigned class labels.
    """
    labels = torch.tensor(dataset.targets)
    num_classes = len(labels.unique())

    # 1. Create a list of data indices for each class and shuffle them
    classes_list = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels):
        classes_list[label.item()].append(idx)
    for class_indices in classes_list:
        np.random.shuffle(class_indices)

    # 2. Determine the Budget
    total_desired_samples = int(
        sum([(min_val + max_val) / 2.0 for min_val, max_val in zip(min_data_num, max_data_num)]))
    samples_per_class = total_desired_samples // num_classes
    remainder = total_desired_samples % num_classes
    target_samples_for_class = [samples_per_class] * num_classes
    for i in range(remainder):
        target_samples_for_class[i] += 1
    #print(f"Total desired samples: {total_desired_samples}. Taking ~{samples_per_class} samples per class.")

    # 3. Split classes and clients into Bipartite Groups
    all_class_indices = np.arange(num_classes)
    np.random.shuffle(all_class_indices)
    num_common_classes = num_classes // 2
    common_classes = all_class_indices[:num_common_classes]
    rare_classes = all_class_indices[num_common_classes:]

    all_client_indices = np.arange(num_users)
    np.random.shuffle(all_client_indices)
    num_majority_clients = int(0.9 * num_users) # only updated for jongik server
    majority_client_ids = all_client_indices[:num_majority_clients]
    minority_client_ids = all_client_indices[num_majority_clients:]

    # 4. === NEW: For each client, select a k-class subset from their group's pool ===
    clients_label_map = {}
    for client_id in majority_client_ids:
        num_classes_for_client = max(1, int(class_ratio * len(common_classes)))
        clients_label_map[client_id] = np.random.choice(common_classes, size=num_classes_for_client, replace=False)
    for client_id in minority_client_ids:
        num_classes_for_client = max(1, int(class_ratio * len(rare_classes)))
        clients_label_map[client_id] = np.random.choice(rare_classes, size=num_classes_for_client, replace=False)

    # 5. Allocate the Budget Proportionally
    client_proportions = np.array([(min_val + max_val) / 2.0 for min_val, max_val in zip(min_data_num, max_data_num)])
    num_samples_per_client = np.zeros(num_users, dtype=int)

    # ... (The proportional allocation logic for num_samples_per_client remains the same as before)
    # ... I am omitting it here for brevity but it should be copied from the previous answer
    # Define the exact data pools based on the budget
    total_data_in_common_pool = sum(target_samples_for_class[c] for c in common_classes)
    total_data_in_rare_pool = sum(target_samples_for_class[c] for c in rare_classes)

    # Allocate for Majority Group
    proportions_majority = client_proportions[majority_client_ids]
    total_proportion_majority = np.sum(proportions_majority)
    assigned_so_far = 0
    if total_proportion_majority > 0:
        for i, client_id in enumerate(majority_client_ids[:-1]):
            share = proportions_majority[i] / total_proportion_majority
            num_samples = int(share * total_data_in_common_pool)
            num_samples_per_client[client_id] = num_samples
            assigned_so_far += num_samples
        if len(majority_client_ids) > 0:
            last_client_id = majority_client_ids[-1]
            num_samples_per_client[last_client_id] = total_data_in_common_pool - assigned_so_far

    # Allocate for Minority Group
    proportions_minority = client_proportions[minority_client_ids]
    total_proportion_minority = np.sum(proportions_minority)
    assigned_so_far = 0
    if total_proportion_minority > 0:
        if len(minority_client_ids) > 1:
            for i, client_id in enumerate(minority_client_ids[:-1]):
                share = proportions_minority[i] / total_proportion_minority
                num_samples = int(round(share * total_data_in_rare_pool))
                num_samples_per_client[client_id] = num_samples
                assigned_so_far += num_samples
            last_client_id = minority_client_ids[-1]
            num_samples_per_client[last_client_id] = total_data_in_rare_pool - assigned_so_far
        elif len(minority_client_ids) == 1:
            num_samples_per_client[minority_client_ids[0]] = total_data_in_rare_pool

    # 6. Distribute Data
    clients_data_idx = [[] for _ in range(num_users)]
    class_data_pointers = np.zeros(num_classes, dtype=int)
    for i in range(num_users):
        num_samples_for_client = int(num_samples_per_client[i])
        available_classes = clients_label_map[i]

        class_cursor = 0
        if len(available_classes) == 0:
            continue

        for _ in range(num_samples_for_client):
            current_class = available_classes[class_cursor]

            # print(f"Error: class {current_class}. {class_data_pointers[current_class]}")

            data_idx = classes_list[current_class][class_data_pointers[current_class]]
            clients_data_idx[i].append(data_idx)
            class_data_pointers[current_class] += 1
            class_cursor = (class_cursor + 1) % len(available_classes)

    return clients_data_idx, clients_label_map