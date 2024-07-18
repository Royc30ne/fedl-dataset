import numpy as np
import os
from collections import defaultdict
import argparse
from utils import raw_data_util
from tqdm import tqdm

# Paths to the raw dataset files
RAW_DATA_DIR = './raw_data'
FEDL_DATA_DIR = './fedl_data'

def create_federated_data_iid(train_x, train_y, num_clients, client_id_prefix='client_', random_seed=1):
    num_classes = len(np.unique(train_y))
    federated_data = defaultdict(lambda: {'x': [], 'y': []})
    
    print(f"Creating IID federated data for {num_clients} clients...")
    print(f"Generating {num_clients} clients with {num_classes} classes each.")
    
    for c in range(num_classes):
        
        print(f"Generate class {c} / {num_classes-1}")
        class_indices = np.where(train_y == c)[0]
        np.random.seed(random_seed)
        np.random.shuffle(class_indices)
        class_chunks = np.array_split(class_indices, num_clients)
        
        for i in tqdm(range(num_clients), desc='Clients'):
            client_id = f"{client_id_prefix}{i}"
            federated_data[client_id]['x'].extend(train_x[class_chunks[i]])
            federated_data[client_id]['y'].extend(train_y[class_chunks[i]])
    
    for client_id in federated_data:
        federated_data[client_id]['x'] = np.array(federated_data[client_id]['x'])
        federated_data[client_id]['y'] = np.array(federated_data[client_id]['y'])
    
    return federated_data

def create_federated_data_non_iid(train_x, train_y, num_clients, alpha=0.5, client_id_prefix='client_', random_seed=1):
    federated_data = defaultdict(dict)
    num_classes = len(np.unique(train_y))
    class_indices = [np.where(train_y == i)[0] for i in range(num_classes)]
    
    client_data_indices = [[] for _ in range(num_clients)]
    for idx, c in enumerate(class_indices):
        print(f"Generate class {idx} / {num_classes-1}")
        np.random.seed(random_seed)
        np.random.shuffle(c)
        np.random.seed(random_seed)
        client_distribution = np.random.dirichlet([alpha] * num_clients)
        client_distribution = (client_distribution * len(c)).astype(int)
        np.add.at(client_distribution, np.argmax(client_distribution), len(c) - np.sum(client_distribution))  # adjust the total

        start = 0
        for client, count in tqdm(enumerate(client_distribution), desc='Clients'):
            client_data_indices[client].extend(c[start:start+count])
            start += count

    for i, indices in enumerate(client_data_indices):
        client_id = f"{client_id_prefix}{i}"
        federated_data[client_id]['x'] = train_x[indices]
        federated_data[client_id]['y'] = train_y[indices]

    return federated_data

def save_federated_data(federated_data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for client_id, data in federated_data.items():
        np.save(os.path.join(path, f"{client_id}_x.npy"), data['x'])
        np.save(os.path.join(path, f"{client_id}_y.npy"), data['y'])
    print(f"Federated data saved to {path}")

def save_test_data(test_images, test_labels, path):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, "test_images.npy"), test_images)
    np.save(os.path.join(path, "test_labels.npy"), test_labels)
    print(f"Test data saved to {path}")

def main(args):
    print(f"Preparing federated data for {args.dataset} dataset with {args.num_clients} clients using {args.sample} sampling method.")  
    if args.dataset == 'emnist':
        raw_data_util.download_and_extract_emnist(RAW_DATA_DIR)        
        (train_x, train_y), (test_x, test_y) = raw_data_util.load_emnist(RAW_DATA_DIR, mode=args.subset)
        train_x = train_x / 255.0
        test_x = test_x / 255.0
        
    print(f"{args.dataset} dataset loaded with {len(train_x)} training samples and {len(test_x)} test samples.")
    
    if args.sample == 'iid':
        federated_data = create_federated_data_iid(train_x, train_y, args.num_clients, client_id_prefix=args.c_prefix, random_seed=args.seed)
    else:
        federated_data = create_federated_data_non_iid(train_x, train_y, args.num_clients, alpha=args.alpha, client_id_prefix=args.c_prefix, random_seed=args.seed)
    
    # Save or use federated_data as needed
    save_dir = os.path.join(FEDL_DATA_DIR, f"{args.dataset}_{args.subset}_{args.num_clients}_{args.sample}")
    save_federated_data(federated_data, os.path.join(save_dir, 'train'))
    save_test_data(test_x, test_y, os.path.join(save_dir, 'test'))
    print(f"Federated data created and saved for {len(federated_data)} clients.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning Data Preparation')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (e.g., emnist)')
    parser.add_argument('--subset', type=str, default='balanced', choices=['balanced', 'digits', 'byclass'], help='Subset of the dataset to use')
    parser.add_argument('--num_clients', type=int, default=2000, required=True, help='Number of clients')
    parser.add_argument('-s', '--sample', type=str, required=True, choices=['iid', 'non_iid'], help='Sampling method (iid or non_iid)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for non_iid sample with Dirichlet distribution')
    parser.add_argument('--c_prefix', type=str, default='client_', help='Client name prefix')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator')
    
    main(parser.parse_args())
    
