import os

from utils.dataset_files import get_emnist_files
from huggingface_hub import hf_hub_download


def download_dataset(dataset, subset, data_dir):
    if dataset == 'emnist':
        repo_id = 'Royc30ne/emnist-'+subset
        files = get_emnist_files(subset).values()
        data_dir = os.path.join(data_dir, 'emnist', subset)
        download_from_hugging_face(repo_id, files, data_dir)
    else:
        print(f"Dataset {dataset} not supported.")


def download_from_hugging_face(repo_id, files, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    for file in files:
        print(f"Downloading {file} from huggingface hub...")
        hf_hub_download(repo_id=repo_id,repo_type='dataset', filename=file, local_dir=download_dir)
    print(f"Files downloaded to {download_dir}")
    