import numpy as np
import os
import gzip
import shutil

import utils.download_manager as download_manager

from enum import Enum
from utils.dataset_files import get_emnist_files


class Dataset(Enum):
    EMNIST = 'emnist'


class EMNISTSubset(Enum):
    BALANCED = 'balanced'
    BYCLASS = 'byclass'
    BYMERGE = 'bymerge'
    DIGITS = 'digits'
    LETTERS = 'letters'
    MNIST = 'mnist'
    
    
def extract_gz(file_path):
    try:
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        exit(f"Error extracting {file_path}: {e}")
        
        
def load_images(file_path):
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


def load_raw_data(data_dir, dataset: Dataset, subset: EMNISTSubset):
    if dataset == Dataset.EMNIST:
        return load_emnist(data_dir, subset)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")



def load_emnist(data_dir, subset=EMNISTSubset.BALANCED):
    emnist_path = os.path.join(data_dir, 'emnist', subset)
    files = get_emnist_files(subset)
    
    for file in files.values():
        if not os.path.exists(os.path.join(emnist_path, file)):
            print(f"EMNIST {subset} dataset not found.")
            print(f"Downloading EMNIST {subset} dataset...")
            download_manager.download_dataset('emnist', subset, data_dir)
        
        if not os.path.exists(os.path.join(emnist_path, file[:-3])) & os.path.exists(os.path.join(emnist_path, file)):
            extract_gz(os.path.join(emnist_path, file))

    train_images = load_images(os.path.join(emnist_path, files['train_x'][:-3]))
    train_labels = load_labels(os.path.join(emnist_path, files['train_y'][:-3]))
    test_images = load_images(os.path.join(emnist_path, files['test_x'][:-3]))
    test_labels = load_labels(os.path.join(emnist_path, files['test_y'][:-3]))

    return (train_images, train_labels), (test_images, test_labels)