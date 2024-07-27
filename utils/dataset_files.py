def get_emnist_files(subset):
    files = {
        'train_x': f'emnist-{subset}-train-images-idx3-ubyte.gz',
        'train_y': f'emnist-{subset}-train-labels-idx1-ubyte.gz',
        'test_x': f'emnist-{subset}-test-images-idx3-ubyte.gz',
        'test_y': f'emnist-{subset}-test-labels-idx1-ubyte.gz'
    }
    return files