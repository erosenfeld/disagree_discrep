DATASET_SHIFTS = {
        'entity13': [1, 2, 3],
        'entity30': [1, 2, 3],
        'cifar10': [1, 10, 71, 95],
        'cifar100': [4, 12, 59, 82],
        'domainnet': [1, 2, 3],
        'living17': [1, 2, 3],
        'nonliving26': [1, 2, 3],
        'fmow': [1, 2],
        'officehome': [1, 2, 3],
        'visda': [1, 2],
}

DATASETS = list(DATASET_SHIFTS.keys())

TRAIN_METHODS = [
    'ERM-aug-imagenet',
    'ERM-aug-rand',
    'BN_adapt',
    'DANN',
    'CDANN',
    'FixMatch',
]
PCA_FACTORS = [1, 4, 16, 32, 64, 128]
BOUND_STRATEGIES = [f'PCA{i}' for i in PCA_FACTORS] + ['logits']
