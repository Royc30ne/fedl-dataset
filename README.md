# fedl-dataset

A toolkit for converting centralized datasets into federated learning datasets. Supports EMNIST, CIFAR-10, and CIFAR-100 with multiple partitioning strategies, data poisoning attacks, and automatic distribution statistics.

## Features

- **Download and Extract Dataset**: Automatically downloads and extracts datasets if not already available.
- **Multiple Partitioning Strategies**:
  - **IID**: Uniform random split across clients.
  - **Non-IID (Dirichlet)**: Class-level heterogeneity controlled by α parameter.
  - **Pathological**: Each client gets only K classes (classic FedAvg setting).
  - **Quantity Skew**: IID labels but unequal data sizes per client.
- **Attack Simulation**:
  - **Random Label**: Randomly shuffles labels for attacker clients.
  - **Label Mapping**: Maps one label to another for targeted attacks.
  - **Noise Feature**: Adds Gaussian noise to features.
- **Distribution Statistics**: Automatically generates `stats.json` with per-client sample counts and class distributions.
- **Flexible Client Configuration**: Customizable client ID prefix, random seed, and more.

## Requirements

- Python 3.x
- `numpy`
- `wget`
- `tqdm`
- `huggingface_hub`

Install the required Python packages using:

```sh
pip install -r requirements.txt
```

## Usage

### Basic Partitioning

**IID Sampling**

```sh
python main.py --dataset emnist --subset digits --num_clients 10 -s iid --seed 42
```

**Non-IID Sampling (Dirichlet)**

```sh
# α=0.5 (moderate heterogeneity)
python main.py --dataset emnist --subset digits --num_clients 10 -s non_iid --alpha 0.5 --seed 42

# α=0.1 (high heterogeneity)
python main.py --dataset cifar10 --num_clients 100 -s non_iid --alpha 0.1 --seed 42
```

**Pathological Non-IID (K classes per client)**

```sh
# Each client gets exactly 2 classes (FedAvg paper setting)
python main.py --dataset emnist --subset digits --num_clients 10 -s pathological --n_classes 2 --seed 42

# Each client gets 5 classes
python main.py --dataset cifar100 --num_clients 50 -s pathological --n_classes 5 --seed 42
```

**Quantity Skew (unequal data sizes)**

```sh
python main.py --dataset cifar10 --num_clients 20 -s quantity_skew --alpha 0.5 --seed 42
```

### Attack Simulation

**Random Label Attack**

```sh
python main.py --dataset emnist --subset digits --num_clients 10 -s iid \
    --attack random_label --attack_percentage 0.2 --seed 42
```

**Label Mapping Attack**

```sh
python main.py --dataset emnist --subset digits --num_clients 10 -s iid \
    --attack label_mapping --source_label 0 --target_label 1 --attack_percentage 0.5 --seed 42
```

**Noise Feature Attack**

```sh
python main.py --dataset cifar10 --num_clients 10 -s non_iid --alpha 0.5 \
    --attack noise_feature --noise_std 0.3 --attack_percentage 0.1 --seed 42
```

### Test Data Options

```sh
# Test data kept on server (default)
python main.py --dataset emnist --subset digits --num_clients 10 -s iid --test_owner server

# Test data distributed to clients
python main.py --dataset emnist --subset digits --num_clients 10 -s iid --test_owner client

# Both server and client test data
python main.py --dataset emnist --subset digits --num_clients 10 -s iid --test_owner both
```

## Benchmark Experiment Recipes

| Experiment | Strategy | Key Parameters | Command |
|-----------|----------|---------------|---------|
| FedAvg baseline | `iid` | — | `--dataset cifar10 --num_clients 100 -s iid` |
| FedAvg non-IID | `pathological` | `--n_classes 2` | `--dataset cifar10 --num_clients 100 -s pathological --n_classes 2` |
| Heterogeneity study | `non_iid` | `--alpha 0.1/0.5/1.0` | `--dataset cifar10 --num_clients 100 -s non_iid --alpha 0.1` |
| Data imbalance | `quantity_skew` | `--alpha 0.5` | `--dataset cifar10 --num_clients 100 -s quantity_skew --alpha 0.5` |
| Byzantine robustness | `iid` + attack | `--attack random_label` | `--dataset cifar10 --num_clients 100 -s iid --attack random_label --attack_percentage 0.2` |
| Backdoor attack | `iid` + attack | `--attack label_mapping` | `--dataset cifar10 --num_clients 100 -s iid --attack label_mapping --source_label 0 --target_label 1 --attack_percentage 0.3` |

## Arguments

### Dataset

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--dataset` | str | *(required)* | `emnist`, `cifar10`, `cifar100` | Dataset to use |
| `--subset` | str | `balanced` | `balanced`, `digits`, `byclass` | EMNIST subset |
| `--test_owner` | str | `server` | `server`, `client`, `both` | Where test data is stored |

### Partitioning

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--num_clients` | int | *(required)* | — | Number of clients |
| `-s`, `--sample` | str | *(required)* | `iid`, `non_iid`, `pathological`, `quantity_skew` | Partitioning strategy |
| `--alpha` | float | `0.5` | — | Dirichlet α for `non_iid` / `quantity_skew` (smaller = more heterogeneous) |
| `--n_classes` | int | `2` | — | Classes per client for `pathological` |
| `--balance` / `--no_balance` | flag | `True` | — | Balance client sizes in `non_iid` mode |
| `--c_prefix` | str | `client_` | — | Client ID prefix |
| `--seed` | int | `1` | — | Random seed |

### Attack

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--attack` | str | `None` | `random_label`, `label_mapping`, `noise_feature` | Attack type |
| `--attack_percentage` | float | `0.1` | — | Fraction of clients to attack (0.0 - 1.0) |
| `--source_label` | int | `0` | — | Source label for `label_mapping` |
| `--target_label` | int | `1` | — | Target label for `label_mapping` |
| `--noise_std` | float | `0.5` | — | Gaussian noise std for `noise_feature` |

## Output Structure

```
fedl_data/
└── emnist_digits_10_iid/
    ├── train/
    │   ├── client_0_x.npy
    │   ├── client_0_y.npy
    │   ├── client_1_x.npy
    │   └── ...
    ├── test/
    │   ├── test_images.npy
    │   └── test_labels.npy
    └── stats.json
```

The `stats.json` file contains:
- Per-client sample counts and class distributions
- Summary statistics (mean, std, min, max samples per client)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The EMNIST dataset is provided by the National Institute of Standards and Technology (NIST).
- CIFAR-10 and CIFAR-100 datasets by Alex Krizhevsky.
