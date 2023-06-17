######################################################
# Transfer-learning data loader for the Nôm dataset. #
######################################################

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from torch.utils.data import dataset, sampler, dataloader
import torch
import glob
import numpy as np

from utils import load_image, identity


PRE_TRAIN_PATH = os.environ['_PRE_TRAIN_PATH']
FINE_TUNE_PATH = os.environ['_FINE_TUNE_PATH']

TRAIN_SPLIT_NAME = os.environ['_TRAIN_SPLIT_NAME']
VAL_SPLIT_NAME = os.environ['_VAL_SPLIT_NAME']
TEST_SPLIT_NAME = os.environ['_TEST_SPLIT_NAME']

CHARACTER_FILE_EXTENSION = os.environ['_CHARACTER_FILE_EXTENSION']

PRE_TRAIN_NAME = os.environ["_PRE_TRAIN_NAME"]
FINE_TUNE_NAME = os.environ["_FINE_TUNE_NAME"]


def get_image_class_pairs(character_paths):
    """ Returns a list of image-class pairs.

    Args:
        character_paths (list of str): paths to character directories.
    Returns:
        res (list of tuple of (str, int)): list of image-class pairs.
    """

    res = []

    for class_id, character_path in enumerate(character_paths):
        image_paths = glob.glob(
                os.path.join(character_path, f'*.{CHARACTER_FILE_EXTENSION}')
            )
        class_id = np.repeat(class_id, len(image_paths))
        res.extend(zip(image_paths, class_id))
    
    return res


class NomDataset(dataset.Dataset):
    """ Dataset of Nôm characters."""

    def __init__(self):
        pass


    def __getitem__(self, data_point):
        """Returns a pair of image and its class id.
        
        Args:
            data_point (tuple[str, int]): a data point.
        Returns:
            image (torch.Tensor): image tensor.
            class_id (int): class id of this image tensor.
        """

        file_path, class_id = data_point
        image = load_image(file_path)

        return image, int(class_id)
    

class NomSampler(sampler.Sampler):
    def __init__(self, character_paths, num_images_to_sample):
        """Initializes a NomSampler.

        Args:
            character_paths (list of str): paths to character directories
            num_images_to_sample (int): number of images need to be sampled
        """

        self._data_points = get_image_class_pairs(character_paths)
        self._num_images_to_sample = num_images_to_sample


    def __iter__(self):
        """ Returns an iterator of sampled data points.

        Returns:
            iterator: iterator over task specification keys
        """

        return (
            np.random.default_rng().choice(
                self._data_points,
                size=1,
                replace=False
            )[0] for _ in range(self._num_images_to_sample)
        )

    
    def __len__(self):
        return self._num_images_to_sample


def get_nom_dataloader(phase,
                       split,
                       batch_size,
                       num_images_to_sample):
    """Returns a DataLoader for the Nôm dataset.

    Args:
        phase (str): one of 'pre-train', 'fine-tune'
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of images per batch
        num_images_to_sample (int): number of sampled images before DataLoader exhausted
    Raises:
        ValueError: if phase or split is invalid
    """

    # Determine training phase
    if phase == PRE_TRAIN_NAME:
        data_dir = PRE_TRAIN_PATH
    elif phase == FINE_TUNE_NAME:
        data_dir = FINE_TUNE_PATH
    else:
        raise ValueError('Invalid phase.')

    # Determine split
    if split == 'train':
        data_path = os.path.join(data_dir, TRAIN_SPLIT_NAME)
    elif split == 'val':
        data_path = os.path.join(data_dir, VAL_SPLIT_NAME)
    elif split == 'test':
        data_path = os.path.join(data_dir, TEST_SPLIT_NAME)
    else:
        raise ValueError('Invalid split.')
    
    # Get paths to character directories
    character_paths = sorted(glob.glob(os.path.join(data_path, '*/')))
    
    return dataloader.DataLoader(
        dataset=NomDataset(),
        batch_size=batch_size,
        sampler=NomSampler(character_paths, num_images_to_sample),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
