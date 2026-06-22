# Experiments

`main2.py` is the experiment entrypoint. It parses command-line flags from `utility/parser.py`, builds datasets and models, runs client selection/training rounds, and writes results to `result/<experiment-name>/`.

## Quick Command

```bash
python main2.py \
  --task_type cifar10 \
  --iid_type noniid \
  --algo_type proposed \
  --round_num 150
```

The defaults in `utility/parser.py` are research defaults and may run multiple tasks. For quicker checks, pass a single `--task_type`, matching `--iid_type`, and a smaller `--round_num`.

## Experiment Sweep Script

Use the curated sweep script from the repository root:

```bash
bash scripts/run_experiments.sh
```

The script defines the main sweep variables near the top:

- `seedlist`: random seeds.
- `dlist`: data ratio passed to `--data_ratio`.
- `C`: active communication rate passed to `--C`.
- `task_list`: task names passed to `--task_type`.
- `iid`: IID/non-IID task pattern passed to `--iid_type`.
- `client_n`: number of clients passed to `--num_clients`.
- `class_ratio`: repeated class ratio values passed to `--class_ratio`.

Most commands in the script are preserved as commented alternatives. Uncomment the variant you want to run and keep the notes suffix descriptive, because it becomes part of the output folder name.

## Important Flags

- `--task_type`: one or more tasks such as `cifar10`, `mnist`, `fashion_mnist`, `emnist`, or `shakespeare`.
- `--iid_type`: one entry per task, typically `iid` or `noniid`.
- `--algo_type`: algorithm list, including values such as `proposed`, `random`, `bayesian`, and `round_robin`.
- `--C`: active rate for client communication.
- `--num_clients`: number of clients in the simulation.
- `--local_epochs`: local epoch count per task.
- `--round_num`: number of federated rounds.
- `--notes`: suffix used in the result directory name.
- `--insist`: overwrite an existing result directory for the same run.

## Outputs

Runs write logs and serialized intermediate data under `result/`. That directory is ignored by git so experiment outputs do not clutter source control.

TorchVision datasets are downloaded under `utility/dataset/`. The Shakespeare JSON split is checked in; downloaded vision datasets are ignored.
