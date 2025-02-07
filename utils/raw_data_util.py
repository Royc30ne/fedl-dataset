import numpy as np
import os
import gzip
import wget
import pickle
import shutil
import zipfile


# Dataset URLs
EMNIST_URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'         # EMNIST URL
CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'     # CIFAR-10 URL
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'   # CIFAR-100 URL

# CIFAR-10 dataset file names

# EMNIST dataset file names
EMNIST_BYCLASS_TRAIN_IMAGES = 'emnist-byclass-train-images-idx3-ubyte.gz'
EMNIST_BYCLASS_TRAIN_LABELS = 'emnist-byclass-train-label-idx1-ubyte.gz'
EMNIST_BYCLASS_TEST_IMAGES = 'emnist-byclass-test-images-idx3-ubyte.gz'
EMNIST_BYCLASS_TEST_LABELS = 'emnist-byclass-test-label--idx1-ubyte.gz'
EMNIST_DIGIT_TRAIN_IMAGES = 'emnist-digits-train-images-idx3-ubyte.gz'
EMNIST_DIGIT_TRAIN_LABELS = 'emnist-digits-train-labels-idx1-ubyte.gz'
EMNIST_DIGIT_TEST_IMAGES = 'emnist-digits-test-images-idx3-ubyte.gz'
EMNIST_DIGIT_TEST_LABELS = 'emnist-digits-test-labels-idx1-ubyte.gz'
EMNIST_BALANCED_TRAIN_IMAGES = 'emnist-balanced-train-images-idx3-ubyte.gz'
EMNIST_BALANCED_TRAIN_LABELS = 'emnist-balanced-train-labels-idx1-ubyte.gz'
EMNIST_BALANCED_TEST_IMAGES = 'emnist-balanced-test-images-idx3-ubyte.gz'
EMNIST_BALANCED_TEST_LABELS = 'emnist-balanced-test-labels-idx1-ubyte.gz'

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


def load_emnist(data_path, mode='digits'):
    emnist_path = os.path.join(data_path, 'emnist/gzip')
    if mode == 'digits':
        train_images = os.path.join(emnist_path, EMNIST_DIGIT_TRAIN_IMAGES)
        train_labels = os.path.join(emnist_path, EMNIST_DIGIT_TRAIN_LABELS)
        test_images = os.path.join(emnist_path, EMNIST_DIGIT_TEST_IMAGES)
        test_labels = os.path.join(emnist_path, EMNIST_DIGIT_TEST_LABELS)
    elif mode == 'balanced':
        train_images = os.path.join(emnist_path, EMNIST_BALANCED_TRAIN_IMAGES)
        train_labels = os.path.join(emnist_path, EMNIST_BALANCED_TRAIN_LABELS)
        test_images = os.path.join(emnist_path, EMNIST_BALANCED_TEST_IMAGES)
        test_labels = os.path.join(emnist_path, EMNIST_BALANCED_TEST_LABELS)
    elif mode == 'byclass':
        train_images = os.path.join(emnist_path, EMNIST_BYCLASS_TRAIN_IMAGES)
        train_labels = os.path.join(emnist_path, EMNIST_BYCLASS_TRAIN_LABELS)
        test_images = os.path.join(emnist_path, EMNIST_BYCLASS_TEST_IMAGES)
        test_labels = os.path.join(emnist_path, EMNIST_BYCLASS_TEST_LABELS)
        
    if not os.path.exists(train_images[:-3]):
        extract_gz(train_images)
    if not os.path.exists(train_labels[:-3]):
        extract_gz(train_labels)
    if not os.path.exists(test_images[:-3]):
        extract_gz(test_images)
    if not os.path.exists(test_labels[:-3]):
        extract_gz(test_labels)

    train_images = load_images(train_images[:-3])
    train_labels = load_labels(train_labels[:-3])
    test_images = load_images(test_images[:-3])
    test_labels = load_labels(test_labels[:-3])

    return (train_images, train_labels), (test_images, test_labels)