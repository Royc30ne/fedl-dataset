import numpy as np
import os
import gzip
import json
import wget
import pickle
import shutil
import zipfile

from utils.download_manager import download_dataset


DATASET_FILE_INDEX = './utils/dataset_files.json'
# Dataset URLs
EMNIST_URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'         # EMNIST URL
CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'     # CIFAR-10 URL
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'   # CIFAR-100 URL


def extract_gz(file_path):
    """
    Extracts a .gz file to the same directory
    """
    with gzip.open(file_path, 'rb') as f_in:
        with open(file_path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


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