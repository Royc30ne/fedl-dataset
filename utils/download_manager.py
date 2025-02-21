import os

from utils.dataset_files import get_emnist_files
from huggingface_hub import hf_hub_download


# def download_dataset(dataset, subset, data_dir):
#     if dataset == 'emnist':
#         repo_id = 'Royc30ne/emnist-'+subset
#         files = get_emnist_files(subset).values()
#         data_dir = os.path.join(data_dir, 'emnist', subset)
#         download_from_hugging_face(repo_id, files, data_dir)
#     elif dataset == 'cifar10':
#         print("MNIST dataset is already available in PyTorch.")
#     else:
#         print(f"Dataset {dataset} not supported.")


def download_dataset(dataset, data_dir, subset=None):

    # Check if the dataset already exists
    if subset:
        dataset_path = os.path.join(data_dir, dataset, subset)
    else:  
        dataset_path = os.path.join(data_dir, dataset)
    
    # Create the data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if dataset == 'emnist':
        # emnist must have subset
        if not subset:
            print("EMNIST dataset requires a subset.")
            exit()
        
        try:
            # Try to download the dataset from huggingface hub
            print(f"Downloading {dataset} dataset from huggingface hub...")
            repo_id = 'Royc30ne/emnist-'+subset
            files = get_emnist_files(subset).values()
            data_dir = os.path.join(data_dir, 'emnist', subset)
            download_from_hugging_face(repo_id, files, data_dir)
            

        except Exception as e:
            print(f"Error downloading dataset: {e}")


def download_from_hugging_face(repo_id, files, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    for file in files:
        if os.path.exists(os.path.join(download_dir, file)):
            print(f"File {file} already exists. Skipping download.")
            continue
        print(f"Downloading {file} from huggingface hub...")
        hf_hub_download(repo_id=repo_id,repo_type='dataset', filename=file, local_dir=download_dir)
    print(f"Files downloaded to {download_dir}")
    