# Repository Structure

This repository keeps the original script-based research workflow intact. `main2.py` remains at the root so existing commands and `utility.*` imports continue to work.

## Top Level

- `main2.py`: orchestrates each experiment. It parses arguments, prepares client/task state, loads datasets and models, runs training rounds, aggregates updates, evaluates metrics, and writes results.
- `scripts/run_experiments.sh`: curated sweep commands and commented alternatives from the original experiment script.
- `figures/`: checked-in result figures moved out of the repository root.
- `docs/`: lightweight documentation for running and navigating the code.
- `requirements.txt`: unpinned dependency list derived from imports. Pin versions in a separate environment file when reproducing a specific paper result.

## Utility Modules

- `utility/parser.py`: command-line argument definitions.
- `utility/preprocessing.py`: TorchVision dataset loading and per-client data sizing.
- `utility/dataset.py`: IID and non-IID client data partitioning.
- `utility/load_model.py`: task-name to model factory.
- `utility/model_list.py`: neural network definitions for image and Shakespeare tasks.
- `utility/training.py`: local training loops, alpha-fairness loss, and SCAFFOLD training support.
- `utility/aggregation.py`: federated aggregation, stale update handling, variance-related helpers, and gradient utilities.
- `utility/optimal_sampling.py`: optimal sampling, distribution solvers, stale decay estimation, and CVXPY-based routines.
- `utility/matching.py`: OMP-style client/update matching helpers.
- `utility/taskallocation.py`: task assignment and round-robin allocation helpers.
- `utility/evalation.py`: evaluation, local loss/accuracy, and group fairness metrics.
- `utility/config.py`: task-specific learning-rate defaults.
- `utility/scaffold.py`: SCAFFOLD optimizer implementation.
- `utility/language_tools.py`: Shakespeare dataset loading and sequence utilities.
- `utility/dataset/`: checked-in Shakespeare split plus ignored downloaded TorchVision datasets.

## Runtime Paths

The code uses relative paths:

- `./utility/dataset` for downloaded datasets.
- `./result/<experiment-name>` for run outputs.

Run scripts from the repository root, or use `scripts/run_experiments.sh`, which changes into the root before launching Python.
