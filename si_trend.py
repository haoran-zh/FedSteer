import random

import torchvision.transforms as transforms
import torchvision

transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.EMNIST('./utility/dataset', train=True, download=True,
                                                     transform=transform_train, split='balanced')
testset = torchvision.datasets.EMNIST('./utility/dataset', train=False, download=True,
                                                    transform=transform_test, split='balanced')

input_size = 28
classes_size = 47 # excluding capital letters that look similar to their lowercase counterparts



import torch
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

folder_name_optimal = "./result/1task_nnnnn_class0.3c0.1uvNo_a5.0_OV2jongik_16"
folder_name_efficient = "./result/1task_nnnnn_class0.3c0.1uvNo_a5.0_OV3_16"
with open(folder_name_optimal+"/client_labels_0.pkl", "rb") as f:
    client_labels = pickle.load(f)

# Assume trainset.targets is a torch tensor
labels = trainset.targets

# Step 1: Get unique labels per client
label_distribution = []
for client in range(40):
    # get the labels for this client
    client_indices = client_labels[client]
    client_unique_labels = labels[client_indices].unique().tolist()
    label_distribution.append(client_unique_labels)



# Step 2: Convert list of label sets to binary matrix (multi-hot encoding)
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(label_distribution)  # shape: [40, num_classes]


# Step 3: Apply clustering (e.g., KMeans with 5 clusters)
num_clusters = 8
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(label_matrix)


# print which cluster include which clients
for cluster_id in range(num_clusters):
    print(f"Cluster {cluster_id} includes clients: {np.where(clusters == cluster_id)[0]}")


import pickle
import numpy as np


# read gradient_AS.pkl
import pickle

# show allocation conditions at stars in the plot of optimal beta
# read allocation result
with open(folder_name_efficient+"/s.pkl", "rb") as f:
    si = pickle.load(f)

si = np.array(si)
# compute the average si for clients within the same cluster
cluster_avg = []
for cluster_id in range(num_clusters):
    cluster_clients = np.where(clusters == cluster_id)[0]
    avg = 0
    for i in cluster_clients:
        for j in cluster_clients:
            avg += np.mean(si[:,i,j])
    avg /= len(cluster_clients)**2
    cluster_avg.append(avg)
print('average s_ij within each cluster \n', cluster_avg)
# si shape: 146 (round) 40 40
# plot si
import matplotlib.pyplot as plt
i = 18
j = 25
plt.plot(si[:,i,j])







# read gradient_AS.pkl
import pickle

# show allocation conditions at stars in the plot of optimal beta
# read allocation result
with open(folder_name_optimal+"/s.pkl", "rb") as f:
    si = pickle.load(f)

si = np.array(si)
# compute the average si for clients within the same cluster
cluster_avg = []
for cluster_id in range(num_clusters):
    cluster_clients = np.where(clusters == cluster_id)[0]
    avg = 0
    for i in cluster_clients:
        for j in cluster_clients:
            avg += np.mean(si[:,i,j])
    avg /= len(cluster_clients)**2
    cluster_avg.append(avg)
print('average s_ij within each cluster \n', cluster_avg)
# si shape: 146 (round) 40 40
# plot si
import matplotlib.pyplot as plt

i = 18
j = 25

plt.plot(si[:,i,j])

# save plot as si.png
plt.savefig("si.png")

