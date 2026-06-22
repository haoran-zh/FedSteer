#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

seedlist=(15 16 17 18 19)
a=1
unbalance_value=(0.0)
dlist=(0.2) # data ratio
C=(0.1) # active rate
task_list=("cifar10")
iid="noniid noniid noniid noniid noniid"
client_n=100
class_ratio=0.5
for task_idx in "${task_list[@]}"; do
for uv in "${unbalance_value[@]}"; do
  for d in "${dlist[@]}"; do
    for c in "${C[@]}"; do
    for sd in "${seedlist[@]}"; do

# FedSteer (main command): corrective projection with OMP core set K=10 and lambda=0.5.
python main2.py --improveOMP --OMP --lam 0.5 --K 10 --V_direct --s_slot 0 --norm_candidates --powerfulCNN --givenProb 5.0 --skipOS --stale --stale_b0 1.0 --stale_b 1.0 --venn_list 0.9 0.1 0.0 --freshness --fairness notfair --data_ratio $d --unbalance $uv 1.0 --alpha $a --notes "$task_idx"_d"$d"_class"$class_ratio"c"$c"uvNo_a5.0B0.9_OVlstsq_Only5IOMPk10l0.5_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio $class_ratio $class_ratio $class_ratio $class_ratio $class_ratio --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 155 --insist


### Baseline: random / proposed sampling without FedSteer correction
python main2.py --powerfulCNN --givenProb 5.0 --skipOS --venn_list 0.9 0.1 0.0 --freshness --fairness notfair --data_ratio $d --unbalance $uv 1.0 --alpha $a --notes "$task_idx"_d"$d"_class"$class_ratio"c"$c"uvNo_a5.0B0.7_random_$sd --optimal_sampling --alpha_loss --C $c --num_clients $client_n --class_ratio $class_ratio $class_ratio $class_ratio $class_ratio $class_ratio --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
### Baseline: full participation upper bound
python main2.py --powerfulCNN --L 1 --client_cpu 1.0 0 0 --fullparticipation --venn_list 1.0 0.0 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes "$task_idx"_d"$d"_class"$class_ratio"c"$c"uvNo_a5.0B0.7_full_$sd --C $c --num_clients $client_n --class_ratio $class_ratio $class_ratio $class_ratio $class_ratio $class_ratio --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist

##### Baseline: MIFA, always reuse all cached stale updates
python main2.py --L 1 --givenProb 5.0 --stale --MILA --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes "$task_idx"_d"$d"_class"$class_ratio"c"$c"uvNo_a5.0B0.9_MIFA_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
###### Baseline: SCAFFOLD control variates
python main2.py --L 1 --givenProb 5.0 --scaffold --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 0.1 --alpha $a --notes "$task_idx"_d"$d"_class"$class_ratio"c"$c"uvNo_a5.0B0.9_scaffold_$sd --C $c --num_clients $client_n --class_ratio 0.3 0.3 0.3 0.3 0.3 --iid_type $iid --task_type $task_idx --algo_type random --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist


### Baseline: FedVARP / FedStale-style stale update reuse with beta sweep
python main2.py --powerfulCNN --givenProb 5.0 --approximation --skipOS --adjustoldVR --stale --stale_b0 0.5 --stale_b 0.0 --freshness --noextra_com --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 1.0 --alpha $a --notes "$task_idx"_d"$d"_class"$class_ratio"c"$c"uvNo_a5.0B0.7_b0.5_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio $class_ratio $class_ratio $class_ratio $class_ratio $class_ratio --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist
python main2.py --powerfulCNN --givenProb 5.0 --approximation --skipOS --adjustoldVR --stale --stale_b0 1.0 --stale_b 0.0 --freshness --noextra_com --venn_list 0.9 0.1 0.0 --fairness notfair --data_ratio $d --unbalance $uv 1.0 --alpha $a --notes "$task_idx"_d"$d"_class"$class_ratio"c"$c"uvNo_a5.0B0.7_b1.0_$sd --alpha_loss --optimal_sampling --C $c --num_clients $client_n --class_ratio $class_ratio $class_ratio $class_ratio $class_ratio $class_ratio --iid_type $iid --task_type $task_idx --algo_type proposed --seed $sd --cpumodel --local_epochs 5 5 5 5 5 --round_num 150 --insist

done
done
done
done
done