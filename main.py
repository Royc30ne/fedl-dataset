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
    """
    Create a non-IID federated dataset by distributing samples among clients
    based on a Dirichlet distribution.
    """

    # --- Setup ---
    N = len(train_y)
    M = N // num_clients  # Each client gets exactly M samples; some leftover might be ignored if N not divisible.
    np.random.seed(random_seed)
    
    federated_data = defaultdict(dict)
    
    # Group indices by class
    unique_classes = np.unique(train_y)
    num_classes = len(unique_classes)
    class_indices = [np.where(train_y == c)[0] for c in unique_classes]
    
    # We'll store the indices for each client
    client_data_indices = [[] for _ in range(num_clients)]

    # --- Step 1: Distribute data among clients (non-IID) class-by-class ---
    for c_idx, c in enumerate(unique_classes):
        print(f"Distributing class {c} ({c_idx+1}/{num_classes})")
        indices_for_this_class = class_indices[c_idx].copy()
        np.random.shuffle(indices_for_this_class)  # shuffle in-place

        n_c = len(indices_for_this_class)
        
        # Draw the Dirichlet distribution for this class among clients
        # This returns an array of length `num_clients` that sums to ~1
        class_dist = np.random.dirichlet([alpha] * num_clients)
        
        # Convert fractional distribution to integer counts for this class
        class_counts = (class_dist * n_c).astype(int)
        # Adjust so the sum of class_counts == n_c
        diff = n_c - np.sum(class_counts)
        # Add the leftover to the client with the largest proportion to keep it simple
        if diff > 0:
            class_counts[np.argmax(class_counts)] += diff
        
        # Assign actual indices to each client
        start = 0
        for client_id in range(num_clients):
            count = class_counts[client_id]
            client_data_indices[client_id].extend(
                indices_for_this_class[start:start+count]
            )
            start += count

    # --- Step 2: Balance the allocations so each client ends up with M samples ---
    client_data_indices = balance_client_counts(
        client_data_indices, 
        target_count=M, 
        random_seed=random_seed
    )
    
    # --- Step 3: Build final federated_data dict ---
    for i, indices in enumerate(client_data_indices):
        client_id = f"{client_id_prefix}{i}"
        federated_data[client_id]['x'] = train_x[indices]
        federated_data[client_id]['y'] = train_y[indices]

    return federated_data

def balance_client_counts(client_data_indices, target_count, random_seed=1):
    """
    Given an initial list of index-allocations `client_data_indices`, ensure each
    client has exactly `target_count` samples (by reassigning from overfull to underfull).
    This step can slightly alter the strict Dirichlet distribution but ensures
    balanced dataset sizes.
    """
    np.random.seed(random_seed)
    
    # Convert each client's index list to a set for easier add/remove
    client_sets = [set(idxs) for idxs in client_data_indices]
    lens = np.array([len(s) for s in client_sets])
    
    # Identify which clients are overfull vs. underfull
    overfull = list(np.where(lens > target_count)[0])
    underfull = list(np.where(lens < target_count)[0])
    
    while overfull and underfull:
        i = overfull[0]
        j = underfull[0]
        
        surplus_i = lens[i] - target_count
        deficit_j = target_count - lens[j]
        
        # Move as many as min(surplus_i, deficit_j) from i -> j
        n_transfer = min(surplus_i, deficit_j)
        
        # Pick n_transfer random indices from client i
        idxs_to_move = np.random.choice(list(client_sets[i]), size=n_transfer, replace=False)
        
        # Reassign
        for idx in idxs_to_move:
            client_sets[i].remove(idx)
            client_sets[j].add(idx)
        
        # Update lens
        lens[i] -= n_transfer
        lens[j] += n_transfer
        
        # Update lists
        if lens[i] == target_count:
            overfull.pop(0)
        elif lens[i] < target_count:
            # i might have gone underfull in edge cases
            overfull.pop(0)
            underfull.append(i)
        
        if lens[j] == target_count:
            underfull.pop(0)
        elif lens[j] > target_count:
            # j might have gone overfull in edge cases
            underfull.pop(0)
            overfull.append(j)
    
    # Convert sets back to sorted lists
    balanced_indices = [sorted(list(s)) for s in client_sets]
    return balanced_indices

def create_random_label_attack(data_y, random_seed=1):
    np.random.seed(random_seed)
    return np.random.shuffle(data_y)

def create_label_mapping_attack(data_y, source_label, target_label, percentage, random_seed=1):
    np.random.seed(random_seed)
    source_indices = np.where(data_y == source_label)[0]
    n_attack = int(len(source_indices) * percentage)
    attack_indices = np.random.choice(source_indices, n_attack, replace=False)
    attacked_y = np.array(data_y)
    attacked_y[attack_indices] = target_label
    return attacked_y


def save_federated_data(federated_data, path):
    # Save federated data to disk
    if not os.path.exists(path):
        os.makedirs(path)
    for client_id, data in federated_data.items():
        np.save(os.path.join(path, f"{client_id}_x.npy"), data['x'])
        np.save(os.path.join(path, f"{client_id}_y.npy"), data['y'])
    print(f"Federated data saved to {path}")


def save_test_data(test_images, test_labels, path):
    # Save test data to disk
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, "test_images.npy"), test_images)
    np.save(os.path.join(path, "test_labels.npy"), test_labels)
    print(f"Test data saved to {path}")


def main(args):
    # Download and load the dataset
    print(f"Preparing federated data for {args.dataset} dataset with {args.num_clients} clients using {args.sample} sampling method.")  
    if args.dataset == 'emnist':
        # raw_data_util.download_dataset(RAW_DATA_DIR)        
        (train_x, train_y), (test_x, test_y) = raw_data_util.load_emnist(RAW_DATA_DIR, subset=args.subset)
        # Normalize pixel values
        train_x = train_x / 255.0
        test_x = test_x / 255.0

    elif args.dataset == 'cifar10':
        train_x, train_y, test_x, test_y = raw_data_util.load_cifar(RAW_DATA_DIR, dataset='cifar10')
    
    elif args.dataset == 'cifar100':
        train_x, train_y, test_x, test_y = raw_data_util.load_cifar(RAW_DATA_DIR, dataset='cifar100')

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print(f"Loading {args.dataset} dataset with {len(train_x)} training samples and {len(test_x)} test samples.")
    
    # Generate federated data
    print("Generating federated data...")
    if args.sample == 'iid':
        train_data = create_federated_data_iid(train_x, train_y, args.num_clients, client_id_prefix=args.c_prefix, random_seed=args.seed)
    else:
        train_data = create_federated_data_non_iid(train_x, train_y, args.num_clients, alpha=args.alpha, client_id_prefix=args.c_prefix, random_seed=args.seed)
    
    # train_data looks like this: federated_data[client_id]['x']  
    # Now, start to process attack dataset
    if args.attack != None:

        # First, lets sample the the attacker clients
        attack_number = int(args.num_clients * args.attack_percentage)
        attack_clients = np.random.choice(list(train_data.keys()), attack_number, replace=False)
        
        # attacker clients' ids will be added to one more prefix based on the original client id
        for client_id in attack_clients:
            # Add the attack suffix to the client id
            new_client_id = f"{client_id}_attacker"
            # Add the attacker data to the federated data
            if args.attack == 'random_label':
                attack_xs = train_data[client_id]['x']
                attack_ys = create_random_label_attack(train_data[client_id]['y'], random_seed=args.seed)
                train_data[new_client_id] = {'x': attack_xs, 'y': attack_ys}
            elif args.attack == 'label_mapping':
                raise NotImplementedError("Label mapping attack is not implemented yet.")
            else:
                raise ValueError(f"Unsupported attack type: {args.attack}")
            # Remove the original client id
            del train_data[client_id]
    else:
        print("No attack applied.")


    # Save or use federated_data as needed
    if args.attack:
        save_dir = os.path.join(FEDL_DATA_DIR, f"{args.dataset}_{args.subset}_{args.num_clients}_{args.sample}_{args.attack}_{args.attack_percentage}")
    else:
        save_dir = os.path.join(FEDL_DATA_DIR, f"{args.dataset}_{args.subset}_{args.num_clients}_{args.sample}")
    
    if args.test_owner == 'server':
        save_federated_data(train_data, os.path.join(save_dir, 'train'))
        save_test_data(test_x, test_y, os.path.join(save_dir, 'test'))
    if args.test_owner == 'client':
        test_data = create_federated_data_iid(test_x, test_y, args.num_clients, client_id_prefix=args.c_prefix, random_seed=args.seed)
        save_federated_data(train_data, os.path.join(save_dir, 'train'))
        save_federated_data(test_data, os.path.join(save_dir, 'test'))  
              
    print(f"Federated data created and saved for {len(train_data)} clients.")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Data Preparation')
    parser.add_argument('--dataset', type=str, required=True, choices=['emnist', 'cifar10', 'cifar100'], help='Dataset to use (e.g., emnist)')
    parser.add_argument('--subset', type=str, default='balanced', choices=['balanced', 'digits', 'byclass'], help='Subset of the dataset to use')
    parser.add_argument('--test_owner', type=str, default='server', choices=['server', 'client'],help='Test data owner')
    parser.add_argument('--num_clients', type=int, default=2000, required=True, help='Number of clients')
    parser.add_argument('-s', '--sample', type=str, required=True, choices=['iid', 'non_iid'], help='Sampling method (iid or non_iid)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for non_iid sample with Dirichlet distribution')
    parser.add_argument('--c_prefix', type=str, default='client_', help='Client name prefix')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator')
    parser.add_argument('--attack', type=str, default=None, choices=['random_label', 'label_mapping'], help='Type of attack to apply')
    parser.add_argument('--attack_percentage', type=float, default=0.1, help='Percentage of data to attack')
    parser.add_argument('--source_label', type=int, default=0, help='Source label for label mapping attack')
    parser.add_argument('--target_label', type=int, default=1, help='Target label for label mapping attack')

    main(parser.parse_args())
    
