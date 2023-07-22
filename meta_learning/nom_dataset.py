##################################################
# Meta-learning data loader for the Nôm dataset. #
##################################################

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from torch.utils.data import dataset, sampler, dataloader
import torch
import glob
import numpy as np

from utils import load_image, identity, is_augmented_image


META_TRAIN_PATH = os.environ['_META_TRAIN_PATH']
META_VAL_PATH = os.environ['_META_VAL_PATH']
META_TEST_PATH = os.environ['_META_TEST_PATH']
CHARACTER_FILE_EXTENSION = os.environ['_CHARACTER_FILE_EXTENSION']


def swap_augmented_examples_to_front(file_paths):
    """ Swaps augmented examples to the front of the list.
    
    Args:
        file_paths (list of str): paths to files.
    Returns:
        res (list of str): paths to files with augmented examples at the front.
    """

    res = []
    for file_path in file_paths:
        if is_augmented_image(file_path):
            res.insert(0, file_path)
        else:
            res.append(file_path)
    return res


class NomDataset(dataset.Dataset):
    """ Dataset of Nôm characters."""

    def __init__(self, num_shot, max_num_query):
        """Initializes a NomDataset.

        Args:
            num_shot (int): number of examples per class in the support set
            max_num_query (int): maximum number of examples per class in the
                query set
        """

        super().__init__()
        self._num_shot = num_shot
        self._max_num_query = max_num_query


    def __getitem__(self, sampled_character_paths):
        """Returns a task specification.

        Args:
            sampled_character_paths (list of str): paths to sampled characters
                for a task
        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_shot, c, h, w)
            labels_support (Tensor): task support labels
                shape (num_way * num_shot,)
            images_query (Tensor): task query images
                shape (n, c, h, w)
            labels_query (Tensor): task query labels
                shape (n,)
        """

        images_support, images_query = [], []
        labels_support, labels_query = [], []

        np.random.seed(42)

        for label, character_path in enumerate(sampled_character_paths):
            # Get a class's examples
            all_file_paths = glob.glob(
                os.path.join(character_path, f'*.{CHARACTER_FILE_EXTENSION}')
            )

            # Sample support and query examples
            expected_n_examples = self._num_shot + self._max_num_query
            n_examples = min(expected_n_examples, len(all_file_paths))
            sampled_file_paths = np.random.choice(
                all_file_paths,
                size=n_examples,
                replace=False
            )

            # Swap augmented examples to the front
            # Since we want to use augmented examples for the support set only
            sampled_file_paths = swap_augmented_examples_to_front(sampled_file_paths)

            # Load images
            images = [load_image(file_path) for file_path in sampled_file_paths]

            # Split into support and query sets
            images_support.extend(images[:self._num_shot])
            labels_support.extend([label] * self._num_shot)
            images_query.extend(images[self._num_shot:])
            labels_query.extend([label] * (n_examples - self._num_shot))

        # Aggregate into tensors
        images_support = torch.stack(images_support)
        labels_support = torch.tensor(labels_support)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)
        
        return images_support, labels_support, images_query, labels_query


class NomSampler(sampler.Sampler):
    """ Samples task specification keys for a NomDataset."""

    def __init__(self, character_paths, num_way, num_tasks):
        """Initializes a NomSampler.

        Args:
            character_paths (list of str): paths to character directories
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        
        super().__init__(None)
        self._character_paths = character_paths
        self._num_way = num_way
        self._num_tasks = num_tasks


    def __iter__(self):
        """Returns an iterator over task specification keys.

        Returns:
            iterator: iterator over task specification keys
        """

        np.random.seed(42)
        return (
            np.random.choice(
                self._character_paths,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )
    

    def __len__(self):
        """Returns the number of tasks to sample.

        Returns:
            int: number of tasks to sample
        """
        
        return self._num_tasks


def get_nom_dataloader(split,
                       batch_size,
                       num_way,
                       num_shot,
                       max_num_query,
                       num_tasks):
    """Returns a DataLoader for the Nôm dataset.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_shot (int): number of examples per class in the support/query set
        num_query (int): maximum number of query examples per class
        num_tasks (int): number of tasks before DataLoader exhausted
    """

    if split == 'train':
        data_path = META_TRAIN_PATH
    elif split == 'val':
        data_path = META_VAL_PATH
    elif split == 'test':
        data_path = META_TEST_PATH
    else:
        raise ValueError('Invalid split.')
    
    character_paths = glob.glob(os.path.join(data_path, '*/'))

    return dataloader.DataLoader(
        dataset=NomDataset(num_shot, max_num_query),
        batch_size=batch_size,
        sampler=NomSampler(character_paths, num_way, num_tasks),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
