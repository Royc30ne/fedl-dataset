# fedl-dataset

This tool is designed to process centralized datasets into federated learning datasets. Currently, it supports the EMNIST dataset but is built to be extended to support other datasets in the future. The tool allows the user to create both IID (Independent and Identically Distributed) and non-IID datasets suitable for federated learning experiments. Additionally, it supports applying random label attacks and label mapping attacks to the data.

## Features

- **Download and Extract Dataset**: Automatically downloads and extracts the EMNIST dataset if not already available.
- **IID Data Partitioning**: Splits the dataset into IID partitions across multiple clients.
- **Non-IID Data Partitioning**: Splits the dataset into non-IID partitions using Dirichlet distribution across multiple clients.
- **Flexible Client Configuration**: Allows custom prefix for client IDs and configurable alpha value for Dirichlet distribution.
- **Random Label Attack**: Randomly shuffles the labels of a specified percentage of the dataset.
- **Label Mapping Attack**: Maps one label to another label for a specified percentage of the dataset.

## Requirements

- Python 3.x
- `numpy`
- `wget`
- `argparse`
- `tqdm`
- `gzip`
- `zipfile`
- `shutil`

Install the required Python packages using:

```sh
pip install numpy wget argparse argparse tqdm gzip zipfile shutil
```

## Usage

**IID Sampling**

```sh
python main.py --dataset emnist --num_clients 10 -s iid --c_prefix client_
```
**Non-IID Sampling**

```sh
python main.py --dataset emnist --num_clients 10 -s noniid --c_prefix client_ --alpha 0.5
```

**Random Label Attack**

```sh
python main.py --dataset emnist --subset balanced --num_clients 10 -s iid --c_prefix client_ --attack random_label --attack_percentage 0.1
```

**Label Mapping Attack**

```sh
python main.py --dataset emnist --subset balanced --num_clients 10 -s iid --c_prefix client_ --attack label_mapping --source_label 0 --target_label 1 --attack_percentage 0.1
```

## Arguments

The following arguments can be used to customize the behavior of the script. 

- `--dataset`: Specifies the dataset to use.
  - **Type**: `str`
  - **Required**: Yes
  - **Example**: `--dataset emnist`
  - **Description**: Dataset to use (e.g., emnist)

- `--subset`: Specifies the subset of the dataset to use.
  - **Type**: `str`
  - **Required**: No
  - **Default**: `balanced`
  - **Choices**: `balanced`, `digits`, `byclass`
  - **Example**: `--subset digits`
  - **Description**: Subset of the dataset to use

- `--test_owner`: Specifies the owner of the test data.
  - **Type**: `str`
  - **Required**: No
  - **Default**: `server`
  - **Choices**: `server`, `client`
  - **Example**: `--test_owner client`
  - **Description**: Test data owner. 
    - When `test_owner` is `server`, the test dataset will be a centralized dataset with train data and train labels.
    - When `test_owner` is `client`, the test data will be sampled to each client, matching the train client_id.

- `--num_clients`: Specifies the number of clients.
  - **Type**: `int`
  - **Required**: Yes
  - **Default**: `2000`
  - **Example**: `--num_clients 1000`
  - **Description**: Number of clients

- `-s, --sample`: Specifies the sampling method to use.
  - **Type**: `str`
  - **Required**: Yes
  - **Choices**: `iid`, `non_iid`
  - **Example**: `--sample iid`
  - **Description**: Sampling method (iid or non_iid)

- `--alpha`: Specifies the alpha value for non_iid sample with Dirichlet distribution.
  - **Type**: `float`
  - **Required**: No
  - **Default**: `0.5`
  - **Example**: `--alpha 0.3`
  - **Description**: Alpha value for non_iid sample with Dirichlet distribution

- `--c_prefix`: Specifies the client name prefix.
  - **Type**: `str`
  - **Required**: No
  - **Default**: `client_`
  - **Example**: `--c_prefix user_`
  - **Description**: Client name prefix

- `--seed`: Specifies the seed for the random number generator.
  - **Type**: `int`
  - **Required**: No
  - **Default**: `1`
  - **Example**: `--seed 42`
  - **Description**: Seed for random number generator

- `--attack`: Specifies the type of attack to apply.
  - **Type**: `str`
  - **Required**: No
  - **Default**: `None`
  - **Choices**: `random_label`, `label_mapping`
  - **Example**: `--attack random_label`
  - **Description**: Type of attack to apply

- `--attack_percentage`: Specifies the percentage of data to attack.
  - **Type**: `float`
  - **Required**: No
  - **Default**: `0.1`
  - **Example**: `--attack_percentage 0.2`
  - **Description**: Percentage of data to attack

- `--source_label`: Specifies the source label for label mapping attack.
  - **Type**: `int`
  - **Required**: No
  - **Default**: `0`
  - **Example**: `--source_label 3`
  - **Description**: Source label for label mapping attack

- `--target_label`: Specifies the target label for label mapping attack.
  - **Type**: `int`
  - **Required**: No
  - **Default**: `1`
  - **Example**: `--target_label 5`
  - **Description**: Target label for label mapping attack


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The EMNIST dataset is provided by the National Institute of Standards and Technology (NIST).
