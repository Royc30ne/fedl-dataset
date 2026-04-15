import numpy as np
import os
import json
from collections import defaultdict
import argparse
from utils import raw_data_util
from tqdm import tqdm

# Paths to the raw dataset files
RAW_DATA_DIR = './raw_data'
FEDL_DATA_DIR = './fedl_data'


# =============================================================================
# Partitioning Strategies
# =============================================================================

def create_federated_data_iid(train_x, train_y, num_clients, client_id_prefix='client_', random_seed=1):
    """
    Create an IID federated dataset by uniformly distributing each class
    across all clients.
    """
    np.random.seed(random_seed)  # Set seed once at entry

    num_classes = len(np.unique(train_y))
    federated_data = defaultdict(lambda: {'x': [], 'y': []})
    
    print(f"Creating IID federated data for {num_clients} clients...")
    print(f"Generating {num_clients} clients with {num_classes} classes each.")
    
    for c in range(num_classes):
        print(f"Generate class {c} / {num_classes-1}")
        class_indices = np.where(train_y == c)[0]
        # BUG #2 FIX: seed is set once above; not reset per class
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


def create_federated_data_non_iid(train_x, train_y, num_clients, alpha=0.5,
                                   client_id_prefix='client_', random_seed=1,
                                   balance=True):
    """
    Create a non-IID federated dataset by distributing samples among clients
    based on a Dirichlet distribution.

    Args:
        train_x: Training features.
        train_y: Training labels.
        num_clients: Number of clients.
        alpha: Concentration parameter for Dirichlet distribution.
               Smaller alpha → more heterogeneous.
        client_id_prefix: Prefix for client IDs.
        random_seed: Random seed for reproducibility.
        balance: If True, balance client data sizes to be equal.
    """
    # --- Setup ---
    N = len(train_y)
    M = N // num_clients  # Target samples per client (used only if balance=True)
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

    # --- Step 2: Optionally balance the allocations ---
    if balance:
        client_data_indices = balance_client_counts(
            client_data_indices, 
            target_count=M,
        )
    
    # --- Step 3: Build final federated_data dict ---
    for i, indices in enumerate(client_data_indices):
        client_id = f"{client_id_prefix}{i}"
        federated_data[client_id]['x'] = train_x[indices]
        federated_data[client_id]['y'] = train_y[indices]

    return federated_data


def create_federated_data_pathological(train_x, train_y, num_clients, n_classes=2,
                                        client_id_prefix='client_', random_seed=1):
    """
    Create a pathological non-IID federated dataset where each client
    receives data from only `n_classes` classes. This is the classic
    non-IID setting from the original FedAvg paper (McMahan et al., 2017).

    The data is first sorted by label, divided into `num_clients * n_classes`
    shards, and each client receives exactly `n_classes` shards.

    Args:
        train_x: Training features.
        train_y: Training labels.
        num_clients: Number of clients.
        n_classes: Number of classes per client (default 2).
        client_id_prefix: Prefix for client IDs.
        random_seed: Random seed for reproducibility.
    """
    np.random.seed(random_seed)

    num_total_classes = len(np.unique(train_y))
    if n_classes > num_total_classes:
        raise ValueError(
            f"n_classes ({n_classes}) cannot exceed the total number of "
            f"classes ({num_total_classes}) in the dataset."
        )

    federated_data = defaultdict(dict)

    # Sort data by label
    sorted_indices = np.argsort(train_y)

    # Divide into shards
    num_shards = num_clients * n_classes
    shard_size = len(train_y) // num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

    # Randomly assign n_classes shards to each client
    shard_indices = np.arange(num_shards)
    np.random.shuffle(shard_indices)

    for i in range(num_clients):
        client_id = f"{client_id_prefix}{i}"
        client_shards = shard_indices[i * n_classes:(i + 1) * n_classes]
        client_indices = np.concatenate([shards[s] for s in client_shards])
        federated_data[client_id]['x'] = train_x[client_indices]
        federated_data[client_id]['y'] = train_y[client_indices]

    return federated_data


def create_federated_data_quantity_skew(train_x, train_y, num_clients, alpha=0.5,
                                         client_id_prefix='client_', random_seed=1):
    """
    Create a federated dataset with quantity skew (IID labels but unequal
    data sizes). The total amount of data per client follows a Dirichlet
    distribution. Within each client, data is sampled uniformly at random
    (preserving the global class distribution).

    Args:
        train_x: Training features.
        train_y: Training labels.
        num_clients: Number of clients.
        alpha: Concentration parameter. Smaller → more imbalanced.
        client_id_prefix: Prefix for client IDs.
        random_seed: Random seed for reproducibility.
    """
    np.random.seed(random_seed)

    N = len(train_y)
    federated_data = defaultdict(dict)

    # Draw data quantity proportions from Dirichlet
    proportions = np.random.dirichlet([alpha] * num_clients)
    client_counts = (proportions * N).astype(int)

    # Adjust so total equals N
    diff = N - np.sum(client_counts)
    if diff > 0:
        client_counts[np.argmax(client_counts)] += diff
    elif diff < 0:
        # Remove excess from the largest client
        client_counts[np.argmax(client_counts)] += diff  # diff is negative

    # Shuffle all indices and distribute
    all_indices = np.arange(N)
    np.random.shuffle(all_indices)

    start = 0
    for i in range(num_clients):
        client_id = f"{client_id_prefix}{i}"
        count = client_counts[i]
        client_indices = all_indices[start:start + count]
        federated_data[client_id]['x'] = train_x[client_indices]
        federated_data[client_id]['y'] = train_y[client_indices]
        start += count

    return federated_data


def balance_client_counts(client_data_indices, target_count):
    """
    Given an initial list of index-allocations `client_data_indices`, ensure each
    client has exactly `target_count` samples (by reassigning from overfull to underfull).
    This step can slightly alter the strict Dirichlet distribution but ensures
    balanced dataset sizes.
    """
    # BUG #5 FIX: Do NOT re-seed here; caller manages the RNG state.

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


# =============================================================================
# Attack Strategies
# =============================================================================

def create_random_label_attack(data_y, random_seed=1):
    """
    Randomly shuffle labels. Returns a new shuffled copy.
    """
    np.random.seed(random_seed)
    # BUG #1 FIX: np.random.shuffle is in-place and returns None.
    # We must copy first, shuffle, then return the copy.
    shuffled_y = data_y.copy()
    np.random.shuffle(shuffled_y)
    return shuffled_y


def create_label_mapping_attack(data_y, source_label, target_label, percentage, random_seed=1):
    """
    Map `source_label` to `target_label` for a given percentage of
    source-label samples.
    """
    np.random.seed(random_seed)
    source_indices = np.where(data_y == source_label)[0]
    n_attack = int(len(source_indices) * percentage)
    attack_indices = np.random.choice(source_indices, n_attack, replace=False)
    attacked_y = data_y.copy()
    attacked_y[attack_indices] = target_label
    return attacked_y


def create_noise_feature_attack(data_x, noise_std=0.5, random_seed=1):
    """
    Add Gaussian noise to features. Returns a new noised copy.

    Args:
        data_x: Feature array.
        noise_std: Standard deviation of the Gaussian noise.
        random_seed: Random seed for reproducibility.
    """
    np.random.seed(random_seed)
    noised_x = data_x.copy().astype(np.float64)
    noise = np.random.normal(0, noise_std, size=noised_x.shape)
    noised_x += noise
    return noised_x


# =============================================================================
# Statistics
# =============================================================================

def compute_distribution_stats(federated_data):
    """
    Compute per-client data distribution statistics.

    Returns a dict with per-client info and a global summary.
    """
    stats = {
        'clients': {},
        'summary': {}
    }

    all_counts = []
    for client_id, data in federated_data.items():
        labels = data['y']
        if labels is None:
            stats['clients'][client_id] = {'num_samples': 0, 'class_distribution': {}}
            all_counts.append(0)
            continue
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
        num_samples = int(len(labels))
        stats['clients'][client_id] = {
            'num_samples': num_samples,
            'class_distribution': class_dist
        }
        all_counts.append(num_samples)

    all_counts = np.array(all_counts)
    stats['summary'] = {
        'num_clients': len(federated_data),
        'total_samples': int(np.sum(all_counts)),
        'samples_per_client': {
            'mean': float(np.mean(all_counts)),
            'std': float(np.std(all_counts)),
            'min': int(np.min(all_counts)),
            'max': int(np.max(all_counts)),
        }
    }
    return stats


def print_stats_summary(stats):
    """Print a concise summary of dataset distribution statistics."""
    s = stats['summary']
    print("\n" + "=" * 60)
    print("  Dataset Distribution Statistics")
    print("=" * 60)
    print(f"  Total clients:          {s['num_clients']}")
    print(f"  Total samples:          {s['total_samples']}")
    print(f"  Samples/client (mean):  {s['samples_per_client']['mean']:.1f}")
    print(f"  Samples/client (std):   {s['samples_per_client']['std']:.1f}")
    print(f"  Samples/client (min):   {s['samples_per_client']['min']}")
    print(f"  Samples/client (max):   {s['samples_per_client']['max']}")
    print("=" * 60 + "\n")


# =============================================================================
# I/O
# =============================================================================

def save_federated_data(federated_data, path):
    """Save federated data to disk."""
    if not os.path.exists(path):
        os.makedirs(path)
    for client_id, data in federated_data.items():
        np.save(os.path.join(path, f"{client_id}_x.npy"), data['x'])
        np.save(os.path.join(path, f"{client_id}_y.npy"), data['y'])
    print(f"Federated data saved to {path}")


def save_test_data(test_images, test_labels, path):
    """Save test data to disk."""
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, "test_images.npy"), test_images)
    np.save(os.path.join(path, "test_labels.npy"), test_labels)
    print(f"Test data saved to {path}")


def save_stats(stats, path):
    """Save distribution statistics as JSON."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {path}")


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Download and load the dataset
    print(f"Preparing federated data for {args.dataset} dataset with "
          f"{args.num_clients} clients using {args.sample} sampling method.")

    if args.dataset == 'emnist':
        (train_x, train_y), (test_x, test_y) = raw_data_util.load_emnist(
            RAW_DATA_DIR, subset=args.subset
        )
        # Normalize pixel values
        train_x = train_x / 255.0
        test_x = test_x / 255.0

    elif args.dataset == 'cifar10':
        train_x, train_y, test_x, test_y = raw_data_util.load_cifar(
            RAW_DATA_DIR, dataset='cifar10'
        )
        train_x = train_x / 255.0
        test_x = test_x / 255.0

    elif args.dataset == 'cifar100':
        train_x, train_y, test_x, test_y = raw_data_util.load_cifar(
            RAW_DATA_DIR, dataset='cifar100'
        )
        # BUG #6 FIX: CIFAR-100 was missing normalization
        train_x = train_x / 255.0
        test_x = test_x / 255.0

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print(f"Loaded {args.dataset} dataset with {len(train_x)} training samples "
          f"and {len(test_x)} test samples.")
    
    # --- Generate federated data ---
    print("Generating federated data...")
    if args.sample == 'iid':
        train_data = create_federated_data_iid(
            train_x, train_y, args.num_clients,
            client_id_prefix=args.c_prefix, random_seed=args.seed
        )
    elif args.sample == 'non_iid':
        train_data = create_federated_data_non_iid(
            train_x, train_y, args.num_clients,
            alpha=args.alpha, client_id_prefix=args.c_prefix,
            random_seed=args.seed, balance=args.balance
        )
    elif args.sample == 'pathological':
        train_data = create_federated_data_pathological(
            train_x, train_y, args.num_clients,
            n_classes=args.n_classes, client_id_prefix=args.c_prefix,
            random_seed=args.seed
        )
    elif args.sample == 'quantity_skew':
        train_data = create_federated_data_quantity_skew(
            train_x, train_y, args.num_clients,
            alpha=args.alpha, client_id_prefix=args.c_prefix,
            random_seed=args.seed
        )
    else:
        raise ValueError(f"Unsupported sampling method: {args.sample}")

    # --- Apply attack (if any) ---
    # BUG #9 FIX: use `is not None` instead of `!= None`
    if args.attack is not None:
        # BUG #3 FIX: set seed before sampling attackers for reproducibility
        np.random.seed(args.seed)
        attack_number = int(args.num_clients * args.attack_percentage)
        attack_clients = np.random.choice(
            list(train_data.keys()), attack_number, replace=False
        )

        print(f"Applying '{args.attack}' attack to {attack_number} clients...")

        # BUG #8 FIX: collect modifications first, apply after loop
        modifications = {}  # new_client_id -> data dict
        clients_to_remove = []

        for client_id in attack_clients:
            new_client_id = f"{client_id}_attacker"

            if args.attack == 'random_label':
                # BUG #4 FIX: copy features instead of using a reference
                attack_xs = train_data[client_id]['x'].copy()
                attack_ys = create_random_label_attack(
                    train_data[client_id]['y'], random_seed=args.seed
                )
                modifications[new_client_id] = {'x': attack_xs, 'y': attack_ys}

            elif args.attack == 'label_mapping':
                attack_xs = train_data[client_id]['x'].copy()
                attack_ys = create_label_mapping_attack(
                    train_data[client_id]['y'],
                    source_label=args.source_label,
                    target_label=args.target_label,
                    percentage=args.attack_percentage,
                    random_seed=args.seed
                )
                modifications[new_client_id] = {'x': attack_xs, 'y': attack_ys}

            elif args.attack == 'noise_feature':
                attack_xs = create_noise_feature_attack(
                    train_data[client_id]['x'],
                    noise_std=args.noise_std,
                    random_seed=args.seed
                )
                attack_ys = train_data[client_id]['y'].copy()
                modifications[new_client_id] = {'x': attack_xs, 'y': attack_ys}

            else:
                raise ValueError(f"Unsupported attack type: {args.attack}")

            clients_to_remove.append(client_id)

        # Apply all modifications at once
        for cid in clients_to_remove:
            del train_data[cid]
        for new_cid, data in modifications.items():
            train_data[new_cid] = data

        print(f"Attack applied to {len(modifications)} clients.")
    else:
        print("No attack applied.")

    # --- Compute and save statistics ---
    stats = compute_distribution_stats(train_data)
    print_stats_summary(stats)

    # --- Build save directory name ---
    save_name_parts = [args.dataset, args.subset, str(args.num_clients), args.sample]
    if args.sample == 'non_iid':
        save_name_parts.append(f"a{args.alpha}")
    elif args.sample == 'pathological':
        save_name_parts.append(f"c{args.n_classes}")
    elif args.sample == 'quantity_skew':
        save_name_parts.append(f"a{args.alpha}")

    if args.attack:
        save_name_parts.append(args.attack)
        save_name_parts.append(str(args.attack_percentage))

    save_dir = os.path.join(FEDL_DATA_DIR, '_'.join(save_name_parts))

    # --- Save data ---
    save_federated_data(train_data, os.path.join(save_dir, 'train'))
    save_stats(stats, os.path.join(save_dir, 'stats.json'))

    if args.test_owner == 'server':
        save_test_data(test_x, test_y, os.path.join(save_dir, 'test'))
    elif args.test_owner == 'client':
        test_data = create_federated_data_iid(
            test_x, test_y, args.num_clients,
            client_id_prefix=args.c_prefix, random_seed=args.seed
        )
        save_federated_data(test_data, os.path.join(save_dir, 'test'))
    elif args.test_owner == 'both':
        save_test_data(test_x, test_y, os.path.join(save_dir, 'test_server'))
        test_data = create_federated_data_iid(
            test_x, test_y, args.num_clients,
            client_id_prefix=args.c_prefix, random_seed=args.seed
        )
        save_federated_data(test_data, os.path.join(save_dir, 'test'))
    else:
        raise ValueError("Invalid test owner. Choose either 'server', 'client', or 'both'.")

    print(f"Federated data created and saved for {len(train_data)} clients.")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Data Preparation')

    # --- Dataset ---
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['emnist', 'cifar10', 'cifar100'],
                        help='Dataset to use (e.g., emnist)')
    parser.add_argument('--subset', type=str, default='balanced',
                        choices=['balanced', 'digits', 'byclass'],
                        help='Subset of the dataset to use')
    parser.add_argument('--test_owner', type=str, default='server',
                        choices=['server', 'client', 'both'],
                        help='Test data owner')

    # --- Partitioning ---
    parser.add_argument('--num_clients', type=int, default=2000, required=True,
                        help='Number of clients')
    parser.add_argument('-s', '--sample', type=str, required=True,
                        choices=['iid', 'non_iid', 'pathological', 'quantity_skew'],
                        help='Partitioning strategy')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter for non_iid / quantity_skew '
                             '(smaller = more heterogeneous)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes per client for pathological partitioning')
    parser.add_argument('--balance', action='store_true', default=True,
                        help='Balance client data sizes in non_iid mode')
    parser.add_argument('--no_balance', dest='balance', action='store_false',
                        help='Disable balancing in non_iid mode')
    parser.add_argument('--c_prefix', type=str, default='client_',
                        help='Client name prefix')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for random number generator')

    # --- Attack ---
    parser.add_argument('--attack', type=str, default=None,
                        choices=['random_label', 'label_mapping', 'noise_feature'],
                        help='Type of attack to apply')
    parser.add_argument('--attack_percentage', type=float, default=0.1,
                        help='Fraction of clients to attack (0.0 - 1.0)')
    parser.add_argument('--source_label', type=int, default=0,
                        help='Source label for label mapping attack')
    parser.add_argument('--target_label', type=int, default=1,
                        help='Target label for label mapping attack')
    parser.add_argument('--noise_std', type=float, default=0.5,
                        help='Noise std for noise_feature attack')

    main(parser.parse_args())
