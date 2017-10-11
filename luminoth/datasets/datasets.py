# from luminoth.datasets import TFRecordDataset
from luminoth.datasets import TFRecordDatasetTfApi

DATASETS = {
    'tfrecord': TFRecordDatasetTfApi,
}


def get_dataset(dataset_type):
    dataset_type = dataset_type.lower()
    if dataset_type not in DATASETS:
        raise ValueError('"{}" is not a valid dataset_type'
                         .format(dataset_type))

    return DATASETS[dataset_type]
