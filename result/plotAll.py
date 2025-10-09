import numpy as np
import matplotlib.pyplot as plt
from parserplot import ParserArgs
import os
from plotAllocation import plot_allocation
from plotAllocation import simulate_allocation
from plotAllocation import tasklist2clientlist
from plotAllocation import simulate_map
import sys


def plotgraph(x1, x2, x3, x4, y1, y2,y3,y4, z1, z2,z3,z4, ytitle, plot_title):

    x = np.arange(len(x1))
    # Find the min, max, and average across all lines
    min_valuesB = np.minimum.reduce([x1, x2,x3,x4])
    max_valuesB = np.maximum.reduce([x1, x2,x3,x4])
    average_valuesB = (x1 + x2+x3 + x4) /4
    min_valuesRand = np.minimum.reduce([y1, y2,y3,y4])
    max_valuesRand = np.maximum.reduce([y1, y2,y3,y4])
    average_valuesRand = (y1 + y2+y3+y4) /4
    min_valuesRR = np.minimum.reduce([z1, z2,z3,z4])
    max_valuesRR = np.maximum.reduce([z1, z2,z3,z4])
    average_valuesRR = (z1 + z2+z3+z4) / 4

    # min_values = np.minimum.reduce([y1, y2, y3])
    # max_values = np.maximum.reduce([y1, y2, y3])
    # average_values = (y1 + y2 + y3) / 3

    # Fill the area between the min and max with green shading
    # plt.fill_between(x, min_values, max_values, color='green', alpha=0.3, label='Shaded Area')



    # Plot the average line in a darker green color

    plt.clf()

    # Plot the average line in a darker green color
    plt.fill_between(x, min_valuesB, max_valuesB, color='blue', alpha=0.5, label='Alpha-fair Client-Task Allocation')

    plt.fill_between(x, min_valuesRand, max_valuesRand, color='orange', alpha=0.5, label='Random Client-Task Allocation')
    plt.fill_between(x, min_valuesRR, max_valuesRR, color='green', alpha=0.5, label='Round Robin Client-Task Allocation')

    # Plot the average line in a darker green color
    plt.plot(average_valuesRand, color='darkorange', linestyle='--')
    plt.plot(average_valuesRR, color='darkgreen', linestyle='--')
    plt.plot(average_valuesB, color='darkblue', linestyle='--')

    # Set the x-label using the dynamic variable
    plt.ylabel(ytitle,fontsize=14)
    plt.xlabel('Num. Global Iterations',fontsize=14)
    # plt.ylabel('Y-axis')
    plt.title(plot_title,fontsize=14)
    plt.ylim([0.2, 0.55])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    """handles, labels = plt.gca().get_legend_handles_labels()

    # Specify the order of items in the legend
    order = [2, 0, 1]

    # Create the legend with the specified order
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right')"""
    plt.legend(loc='lower right')
    # Show the plot
    plt.savefig(os.path.join(path_plot, f'marie_plot.png'))
    plt.clf()

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


def AvgTimeTaken_1trial(exp_array, numTasks):
    epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    algo_num = exp_array.shape[0]
    epsReachedData = np.zeros((algo_num, numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            for k in range(algo_num):
                indexdata = np.searchsorted(exp_array[k][b, :], epsCheckpoints[a])
                epsReachedData[k, b, a] = indexdata if indexdata < len(exp_array[k][b, :]) else 102
    return np.mean(epsReachedData, axis=1)


def MaxTimeTaken_1trial(exp_array, numTasks):
    epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    algo_num = exp_array.shape[0]
    epsReachedData = np.zeros((algo_num, numTasks, len(epsCheckpoints)))
    for b in range(numTasks):
        for a in range(len(epsCheckpoints)):
            for k in range(algo_num):
                indexdata = np.searchsorted(exp_array[k][b, :], epsCheckpoints[a])
                epsReachedData[k, b, a] = indexdata if indexdata < len(exp_array[k][b, :]) else 102
    return np.max(epsReachedData, axis=1)


parser = ParserArgs()
args = parser.get_args()

numRounds = 300
folder_name = args.plot_folder
path_plot = os.path.join('./result', folder_name)

"""allocation_files = [f for f in os.listdir(path_plot) if f.startswith('Algorithm')]
positions = {}
# plot allocation map
targets = ['bayesian', 'proposed', 'random', 'round_robin']
#targets = ['proposed', 'random', 'round_robin']
for i, f in enumerate(allocation_files):
    for target in targets:
        if target in f:
            positions[target] = i
            break

for i in range(len(targets)):
    allocated_tasks_lists = []
    file_name = os.path.join(path_plot, allocation_files[positions[targets[i]]])
    with open(file_name, 'r') as file:
        tasks_data = ''
        recording = False
        for line in file:
            if "Allocated Tasks:" in line:
                recording = True
                tasks_data += line.split(":", 1)[1].strip()
            elif recording:
                # Check if the line still belongs to 'Allocated Tasks'
                if line.startswith('Task[') or 'Round [' in line:

                    recording = False
                else:
                    tasks_data += ' ' + line.strip()

        if tasks_data:
            # Replace spaces with commas and remove any newlines

            tasks_data = tasks_data.replace(' ', ',').replace('\n', '')
            tasks_data = tasks_data.replace(',,', ',')

            tasks_data = tasks_data.replace('][', '],[')
            tasks_data = '[' + tasks_data + ']'

            # Ensure the string is a valid Python list format
            tasks_list = eval(tasks_data)
            tasks_list = [[int(item) for item in sublist] for sublist in tasks_list]

    plot_allocation(tasks_list, path_plot, numRounds, targets[i])"""


def sort_files(files):
    def extract_numbers(file_name):
        parts = file_name.split('_')
        exp_number = int(parts[3].replace('exp', ''))
        algo_number = int(parts[4].replace('algo', '').split('.')[0])
        return algo_number, exp_number

    return sorted(files, key=extract_numbers)

def load_extra_folder(folder_dict):
    # each folder should only contain 1 algo result_old
    # get keys of the folder_dict, key is the folder name
    files = []
    for key in folder_dict:
        this_path = os.path.join('./result', key)
        files += [os.path.join(this_path, f) for f in os.listdir(this_path) if f.startswith('mcf_i_globalAcc')]
    exp_list = []
    for f in files:
        t = np.load(f)
        t = np.where(t <= 0, 0, t)  # t shape: (task_num, numRounds)
        if t.shape[-1] != numRounds:
            # scale to 120 if it is not 120
            num_samples = 300
            real_length = t.shape[-1]
            # Generate 120 evenly spaced indices from 0 to 299
            indices = np.linspace(0, real_length-1, num_samples, endpoint=False).astype(int)
            # Use these indices to sample from the original array
            t = t[:, indices]
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120
    exp_num = int(exp_array.shape[0] / algo_num)

    return exp_array

# read all files
# find all files starting with mcf
# algo_name = ["bayesian", "alpha-fairness", "random", "round robin","optimal_sampling"]


algo_name = ["a3"]
algo_num  = len(algo_name)
extra_folder = {
    "2task_nnn_0.02unbalance_ms_a3_lr1": "ms_a3",
    "2task_nnn_0.02unbalance_OS_clientfair_a3_lr1": "OS_clientFair_a3",
"2task_nnn_0.02unbalance_OS_taskfair_a3_lr1": "OS_taskFair_a3",
"2task_nnn_0.02unbalance_random_lr1": "random",
    "2task_nnn_0.02unbalance_AS_clientfair_a3_lr0.02": "AS_clientFair_a3",
    "2task_nnn_0.02unbalance_AS_taskfair_a3_lr0.02": "AS_taskFair_a3",
}


extra_array = load_extra_folder(extra_folder)

# seeds
paths = []
paths.append(path_plot)
#paths.append(os.path.join('./result_old', "5task_nnnnn_exp4C1c20d1.5-seed15"))
#paths.append(os.path.join('./result_old', "5task_nnnnn_exp4C1c20d1.5-seed15"))
#paths.append(os.path.join('./result_old', "5task_nnnnn_exp4C1c20d1.5-seed15"))
#paths.append(os.path.join('./result_old', "5task_nnnnn_exp4C1c20d1.5-seed15"))
exp_seeds_array = []
for path_plot in paths:
    files = [f for f in os.listdir(path_plot) if f.startswith('mcf_i_globalAcc')]
    files = sort_files(files)

    exp_list = []
    for f in files:
        t = np.load(os.path.join(path_plot, f))
        t = np.where(t <= 0, 0, t) # t shape: (task_num, numRounds)
        if t.shape[-1] != numRounds:
            # scale to 120 if it is not 120
            num_samples = 300
            real_length = t.shape[-1]
            # Generate 120 evenly spaced indices from 0 to 299
            indices = np.linspace(0, real_length-1, num_samples, endpoint=False).astype(int)
            # Use these indices to sample from the original array
            t = t[:, indices]
        exp_list.append(t)
    exp_array = np.array(exp_list)  # shape 3 5 120
    exp_num = int(exp_array.shape[0] / algo_num)



print(exp_array.shape)

# combine exp_array and extra_array
exp_array = np.concatenate((exp_array, extra_array), axis=0)
# combine algo_name and extra algo name
print(exp_array.shape)
extra_algo_name = extra_folder.values()
algo_name = algo_name + list(extra_algo_name)
algo_num = len(algo_name)


task_num = exp_array.shape[1]
"""simulate_history = []
for i in range(4):
    simulate_history.append(simulate_allocation(exp_array[0], 'bayesian'))
plot_allocation(simulate_history[0], path_plot, numRounds, 'bayesian')
clientList = []
for i in range(4):
    clientList.append(tasklist2clientlist(simulate_history[i], task_num)[2])
simulate_map(clientList, path_plot, 'bayesian', task_index=0, exp_num=4)

simulate_history = []
for i in range(4):
    simulate_history.append(simulate_allocation(exp_array[1], 'proposed'))
plot_allocation(simulate_history[0], path_plot, numRounds, 'proposed')
clientList = []
for i in range(4):
    clientList.append(tasklist2clientlist(simulate_history[i], task_num)[2])
simulate_map(clientList, path_plot, 'proposed', task_index=0, exp_num=4)"""


# plot one by one
tasknum = exp_array.shape[1]
alpha = args.alpha
print(folder_name)

for k in range(algo_num): # algo
    for i in range(tasknum): # task
        plt.plot(np.arange(0, numRounds), exp_array[k][i], label=f'task {i}')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Num. Global Iterations')
    plt.grid(linestyle='--', linewidth=0.5)
    title_name = f'Accuracy of different tasks, {algo_name[k]}' if algo_name[k] != 'alpha-fairness' else f'Accuracy of different tasks, {algo_name[k]}, alpha={alpha}'
    plt.title(title_name)
    plt.savefig(os.path.join(path_plot,f'plot_taskAcc_{algo_name[k]}.png'))
    plt.clf()

plt.rcParams['font.size'] = 12

# average accuracy data and plots
Avg_array = AvgAcc1trial(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), Avg_array[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Average Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Average Accuracy over {tasknum} Tasks')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_avgAcc.png'))
plt.clf()

# min acc data and plots
Min_array = MinAcc1trial(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), Min_array[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Minimum Accuracy')
plt.xlabel('Num. Global Iterations')
#plt.ylim([0.1, 0.5])
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Minimum Accuracy over {tasknum} Tasks')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_minAcc.png'))
plt.clf()


# cascade
cas_array = cascadeAcc(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), cas_array[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Cascade Accuracy')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
plt.title(f'Cascade Accuracy over {tasknum} Tasks')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_casAcc.png'))
plt.clf()

# variance acc data and plots
Var = var1trial(exp_array)
for k in range(algo_num):
    plt.plot(np.arange(0, numRounds), Var[k], label=f'{algo_name[k]}')
plt.legend()
plt.ylabel('Variance')
plt.xlabel('Num. Global Iterations')
plt.grid(linestyle='--', linewidth=0.5)
"""plt.bar(algo_name, np.mean(Var, axis=1), width=0.5)
plt.title(f'Variance over {tasknum} Tasks')
plt.tight_layout()"""
plt.savefig(os.path.join(path_plot,'plot_var.png'))
plt.clf()

# Average time taken
Atime = AvgTimeTaken_1trial(exp_array, numTasks=tasknum)
epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for k in range(algo_num):
    plt.plot(epsCheckpoints, Atime[k], 'o-', label=f'{algo_name[k]}')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken in Num. global iterations')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Average time taken for All Tasks to reach eps')
plt.tight_layout()
plt.savefig(os.path.join(path_plot,'plot_avgTimeTaken.png'))
plt.clf()

# Maximum Time taken
Mtime = MaxTimeTaken_1trial(exp_array, numTasks=tasknum)
epsCheckpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for k in range(algo_num):
    plt.plot(epsCheckpoints, Mtime[k], 'o-', label=f'{algo_name[k]}')
plt.xlabel('Accuracy level eps')
plt.ylabel('Time taken (Num. Global Epochs)')
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.title('Max time for All Tasks to reach eps')
plt.tight_layout()
# plt.rcParams['font.size'] = 18
plt.savefig(os.path.join(path_plot, 'plot_maxTimeTaken.png'))
plt.clf()