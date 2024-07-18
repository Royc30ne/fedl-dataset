# fedl-dataset

This tool is designed to process centralized datasets into federated learning datasets. Currently, it supports the EMNIST dataset but is built to be extended to support other datasets in the future. The tool allows the user to create both IID (Independent and Identically Distributed) and non-IID datasets suitable for federated learning experiments.

## Features

- **DDownload and Extract Dataset**: Automatically downloads and extracts the EMNIST dataset if not already available.
- **IID Data Partitioning**: Splits the dataset into IID partitions across multiple clients.
- **Non-IID Data Partitioning**: Splits the dataset into non-IID partitions using Dirichlet distribution across multiple clients.
- **Flexible Client Configuration**: Allows custom prefix for client IDs and configurable alpha value for Dirichlet distribution.
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

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The EMNIST dataset is provided by the National Institute of Standards and Technology (NIST).