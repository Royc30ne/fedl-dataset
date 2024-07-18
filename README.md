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
python script.py --dataset emnist --subset balanced --num_clients 10 -s iid --c_prefix client_ --attack random_label --attack_percentage 0.1
```

**Label Mapping Attack**

```sh
python script.py --dataset emnist --subset balanced --num_clients 10 -s iid --c_prefix client_ --attack label_mapping --source_label 0 --target_label 1 --attack_percentage 0.1
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The EMNIST dataset is provided by the National Institute of Standards and Technology (NIST).
