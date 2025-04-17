import numpy as np
import os
import gzip
import json
import wget
import pickle
import shutil
import zipfile
import tarfile

from utils.download_manager import download_dataset


DATASET_FILE_INDEX = './utils/dataset_files.json'

# Dataset URLs
EMNIST_URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'         # EMNIST URL
CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'     # CIFAR-10 URL
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'   # CIFAR-100 URL


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_gz(file_path):
    """
    Extracts a .gz file to the same directory
    """
    with gzip.open(file_path, 'rb') as f_in:
        with open(file_path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_tar(file_path, extract_path):
    """
    Extracts a .tar file to the specified directory
    """
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def load_images(file_path):
    """
    Load EMNIST images from a .gz file
    """
    with open(file_path, 'rb') as file:
        file.read(16)  # Skip the magic number and dimensions
        data = np.frombuffer(file.read(), dtype=np.uint8)
        data = data.reshape(-1, 28, 28, 1)
    return data


def load_labels(file_path):
    with open(file_path, 'rb') as file:
        file.read(8)  # Skip the magic number and number of items
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    return labels


def download_cifar10(data_dir):
    """
    Download CIFAR-10 dataset
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cifar10_dir = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    if not os.path.exists(cifar10_dir):
        print("Downloading CIFAR-10 dataset...")
        wget.download(CIFAR10_URL, cifar10_dir)
        print("\nDownload complete.")
    else:
        print("CIFAR-10 dataset already exists. Skipping download.")
        

def download_cifar100(data_dir):
    """
    Download CIFAR-100 dataset
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    cifar100_dir = os.path.join(data_dir, 'cifar-100-python.tar.gz')
    if not os.path.exists(cifar100_dir):
        print("Downloading CIFAR-100 dataset...")
        wget.download(CIFAR100_URL, cifar100_dir)
        print("\nDownload complete.")
    else:
        print("CIFAR-100 dataset already exists. Skipping download.")


def download_and_extract_cifar(data_dir, dataset='cifar10'):
    """
    Download and extract CIFAR dataset
    """
    if dataset == 'cifar10':
        download_cifar10(data_dir)
        cifar_dir = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    elif dataset == 'cifar100':
        download_cifar100(data_dir)
        cifar_dir = os.path.join(data_dir, 'cifar-100-python.tar.gz')
    else:
        raise ValueError("Invalid dataset name. Choose either 'cifar10' or 'cifar100'.")
    
    if not os.path.exists(data_dir):
        print(f"Extracting {dataset} dataset...")
        extract_tar(cifar_dir, data_dir)
        print(f"{dataset} dataset extracted to {data_dir}.")
    else:
        print(f"{dataset} dataset already exists. Skipping extraction.")


def load_cifar(data_path, dataset='cifar10'):
    """
    Load CIFAR dataset
    """
    data_path = os.path.join(data_path, 'cifar')
    if dataset == 'cifar10':
        data_dir = os.path.join(data_path, 'cifar-10-batches-py')
    elif dataset == 'cifar100':
        data_dir = os.path.join(data_path, 'cifar-100-python')
    

    download_and_extract_cifar(data_path, dataset)
    if dataset == 'cifar10':
        return load_cifar10(data_dir)
    elif dataset == 'cifar100':
        return load_cifar100(data_dir)


# Function to load CIFAR-10 dataset
def load_cifar10(file_path):
    """
    Load CIFAR-10 dataset
    """
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Load training data
    for i in range(1, 6):
        batch = unpickle(os.path.join(file_path, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels.append(batch[b'labels'])

    # Concatenate training data and labels
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Load test data
    test_batch = unpickle(os.path.join(file_path, 'test_batch'))
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    return train_data, train_labels, test_data, test_labels


# Function to load CIFAR-10 dataset

def load_cifar100(file_path):
    """
    Load CIFAR-100 dataset
    """
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    # Load training data
    train_batch = unpickle(os.path.join(file_path, 'train'))
    train_data = train_batch[b'data']
    train_labels = np.array(train_batch[b'fine_labels'])

    # Load test data
    test_batch = unpickle(os.path.join(file_path, 'test'))
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'fine_labels'])

    return train_data, train_labels, test_data, test_labels

def download_and_extract_emnist(data_dir):
    zip_path = os.path.join(data_dir, 'gzip.zip')
    extract_path = os.path.join(data_dir, 'emnist')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(extract_path):
        print(f"Downloading EMNIST dataset from {EMNIST_URL}...")
        wget.download(EMNIST_URL, zip_path)
        
        print("\nExtracting EMNIST dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        os.remove(zip_path)
        print("EMNIST dataset downloaded and extracted.")
    else:
        print("EMNIST dataset already exists. Skipping download.")


def load_emnist(data_path, subset='digits'):
    emnist_path = os.path.join(data_path, 'emnist', subset)
    file_index = get_dataset_file_index('emnist', subset)

    train_images = os.path.join(emnist_path, file_index['train_x'])
    train_labels = os.path.join(emnist_path, file_index['train_y'])
    test_images = os.path.join(emnist_path, file_index['test_x'])
    test_labels = os.path.join(emnist_path, file_index['test_y'])
        
    if not os.path.exists(train_images) or not os.path.exists(train_labels) or not os.path.exists(test_images) or not os.path.exists(test_labels):
        print("EMNIST dataset files not found. Try to download the dataset ...")
        download_dataset('emnist', data_path, subset)
    
        print("Extracting EMNIST dataset files...")
        extract_gz(train_images + '.gz')
        extract_gz(train_labels + '.gz')
        extract_gz(test_images + '.gz')
        extract_gz(test_labels + '.gz')

    train_images = load_images(train_images)
    train_labels = load_labels(train_labels)
    test_images = load_images(test_images)
    test_labels = load_labels(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def get_dataset_file_index(dataset, subset=None):
    file_index = json.load(open(DATASET_FILE_INDEX, 'r'))

    try:
        if subset:
            files = file_index[dataset][subset]
        else:
            files = file_index[dataset]
    
    except KeyError:
        print("Dataset not found in download file index.")
        exit()
        
    return dict(files)