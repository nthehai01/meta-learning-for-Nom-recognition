#########################################################################################
# This file contains code for splitting the original dataset into:                      #
#   1. meta-training, meta-validating, and meta-testing sets for meta-learning,         #
#   2. pre-training and fine-tuning sets for transfer learning,                         #
#       each of which is further split into train, val, and test sets.                  #
#                                                                                       #
# First of all, we split the original dataset into two sets:                            #
#   1. The bigger set contains 80% of the character folders,                            #
#   2. The smaller set contains the rest.                                               #
#                                                                                       #
# The bigger set is further split into meta-training, meta-validating for meta-learning #
# or pre-training set for transfer learning.                                            #
# The smaller set is set to be the meta-testing for meta-learning or                    #
# fine-tuning set for transfer learning.                                                #               
#########################################################################################

import os
import glob
import numpy as np
import shutil
import splitfolders


DATA_PATH = os.environ['_PROC_PATH']

CHARACTER_FILE_EXTENSION = os.environ['_CHARACTER_FILE_EXTENSION']
AUGMENTED_IMAGE_ALIAS = os.environ['_AUGMENTED_IMAGE_ALIAS']

META_TRAIN_PATH = os.environ['_META_TRAIN_PATH']
META_VAL_PATH = os.environ['_META_VAL_PATH']
META_TEST_PATH = os.environ['_META_TEST_PATH']

PRE_TRAIN_PATH = os.environ['_PRE_TRAIN_PATH']
PRE_TRAIN_TRAIN_PATH = os.environ['_PRE_TRAIN_TRAIN_PATH']
PRE_TRAIN_VAL_PATH = os.environ['_PRE_TRAIN_VAL_PATH']
PRE_TRAIN_TEST_PATH = os.environ['_PRE_TRAIN_TEST_PATH']

FINE_TUNE_PATH = os.environ['_FINE_TUNE_PATH']
FINE_TUNE_TRAIN_PATH = os.environ['_FINE_TUNE_TRAIN_PATH']
FINE_TUNE_VAL_PATH = os.environ['_FINE_TUNE_VAL_PATH']
FINE_TUNE_TEST_PATH = os.environ['_FINE_TUNE_TEST_PATH']

BIGGER_SET_RATIO = 0.8

META_TRAIN_RATIO = 0.8

TRANSFER_TRAIN_RATIO = 0.5
TRANSFER_VAL_RATIO = 0.2


def get_character_folders(root):
    """ Loads and yields character folders from the root directory.

    Args:
        root (str): the root directory of the dataset.
    Returns:
        A list of character folders.
    """
    
    return glob.glob(os.path.join(root, '*/'))


def filter_character_folders(character_folders, num_imgs):
    """ Throw away character folders with less than num_imgs images.

    Args:
        character_folders (list): list of character folders.
        num_classes (int): minimum number of images accepted for a character.
    Returns:
        res (list): list of character folders.
    """

    func = lambda r: len(os.listdir(r)) >= num_imgs
    res = list(filter(func, character_folders))

    return res


def split_character_folders(character_folders):
    """ Splits character folders into bigger set and smaller set.

    Args:
        character_folders (list): list of character folders.
    Returns:
        bigger_set (list): list of character folders.
        smaller_set (list): list of character folders.
    """

    split_index = int(len(character_folders) * BIGGER_SET_RATIO)
    
    bigger_set = character_folders[:split_index]
    smaller_set = character_folders[split_index:]

    return bigger_set, smaller_set


def move_folders(src, dst):
    """ Moves folders in src to dst.

    Args:
        src (list): list of source folders.
        dst (str): destination folder.
    """

    for src_folder in src:
        folder_name = src_folder.split('/')[-2]
        
        shutil.copytree(
            src_folder, 
            os.path.join(dst, folder_name)
        )


def create_meta_dataset(bigger_set, smaller_set):
    """ Creates meta-training, meta-validating, and meta-testing sets for meta-learning.

    Args:
        bigger_set (list): list of character folders.
        smaller_set (list): list of character folders.
    """

    train_index = int(len(bigger_set) * META_TRAIN_RATIO)

    # Determine splits
    meta_train = bigger_set[:train_index]
    meta_val = bigger_set[train_index:]
    meta_test = smaller_set

    # Create directories
    os.makedirs(META_TRAIN_PATH)
    os.makedirs(META_VAL_PATH)
    os.makedirs(META_TEST_PATH)

    # Move folders
    move_folders(meta_train, META_TRAIN_PATH)
    move_folders(meta_val, META_VAL_PATH)
    move_folders(meta_test, META_TEST_PATH)


def train_val_test_split(train_ratio, val_ratio, is_pre_training=True):
    """ Splits dataset into train, val, and test sets for transfer learning.

    Args:
        train_ratio (float): ratio of training set.
        val_ratio (float): ratio of validation set.
        is_pre_training (bool): whether the split is for pre-training or fine-tuning.
    """

    # Determine phase
    if is_pre_training:
        input_directory = PRE_TRAIN_PATH
    else:
        input_directory = FINE_TUNE_PATH

    # Split dataset
    test_ratio = 1 - train_ratio - val_ratio
    splitfolders.ratio(
        input_directory, # The location of dataset
        output=input_directory, # The output location
        seed=0, # The number of seed
        ratio=(train_ratio, val_ratio, test_ratio), # The ratio of split dataset
        group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
        move=True # If you choose to move, turn this into True
    )

    # Clean up
    clean_up(input_directory)


def clean_up(dir):
    character_folders = glob.glob(
        os.path.join(dir, '*/')
    )

    for character_path in character_folders:
        character_folder_name = character_path.split('/')[-2]
        if character_folder_name in ['train', 'val', 'test']:
            continue
        else:
            shutil.rmtree(character_path)


def is_augmented_image(file):
    """ Checks if a file is an augmented image by its name.
        A file is an augmented image if its name is of format <name>.aug.<extension>.
        Hence, if we split the file's name by '.', 
        the length of the list should be equal to 3.

    Args:
        file (str): the file's name.
    """

    file_name = file.split('/')[-1]
    return len(file_name.split('.')) == 3


def get_augmented_images(character_folder):
    """ Finds augmented images in a folder of test set.

    Args:
        character_folder (str): the folder's name.
    Returns:
        aug_images (list): list of augmented image files.
    """

    files = glob.glob(os.path.join(character_folder, f'*.{CHARACTER_FILE_EXTENSION}'))

    # Filter out augmented images
    func = lambda file: is_augmented_image(file)
    aug_images = list(filter(func, files))

    return aug_images


def get_augmented_folders(is_pre_training=True):
    """ Finds names of character folders in test set containing augmented images.

    Args:
        is_pre_training (bool): whether the folder is for pre-training or fine-tuning.
    Returns:
        aug_folders (list): list of character folder's names.
    """

    # Determine directories
    if is_pre_training:
        test_dir = PRE_TRAIN_TEST_PATH
    else:
        test_dir = FINE_TUNE_TEST_PATH

    # Get all character folders in this test set
    character_folders = glob.glob(os.path.join(test_dir, '*/'))

    # Filter out folders with augmented images
    func = lambda folder: len(get_augmented_images(folder)) > 0
    aug_folders = list(filter(func, character_folders))

    # Get character folder names
    func = lambda folder: folder.split('/')[-2]
    aug_folders = list(map(func, aug_folders))

    return aug_folders


def move_files(files, dst):
    """ Moves files to dst.

    Args:
        files (list): list of files.
        dst (str): destination folder.
    """

    for file in files:
        shutil.move(file, dst)


def swap_files(character_folder, is_pre_training=True):
    """ Moves augmented images from test directory to train directory, 
        and move one unaugmented image from train directory to test directory.

    Args:
        character_folder (str): a character folder in test set.
        is_pre_training (bool): whether the swap is for pre-training or fine-tuning.
    """

    # Determine directories
    if is_pre_training:
        train_dir = os.path.join(PRE_TRAIN_TRAIN_PATH, character_folder)
        test_dir = os.path.join(PRE_TRAIN_TEST_PATH, character_folder)
    else:
        train_dir = os.path.join(FINE_TUNE_TRAIN_PATH, character_folder)
        test_dir = os.path.join(FINE_TUNE_TEST_PATH, character_folder)

    # Move augmented images from test directory to train directory
    augmented_images = get_augmented_images(test_dir)
    move_files(augmented_images, train_dir)

    # Move one unaugmented image from train directory to test directory
    unaugmented_images = os.path.join(
        train_dir, 
        str(1).zfill(4) + "." + CHARACTER_FILE_EXTENSION
    )
    move_files([unaugmented_images], test_dir)


def solve_augmentation_issue(is_pre_training=True):
    """ Finds and solves augmented images issue in test set.

    Args:
        is_pre_training (bool): whether the swap is for pre-training or fine-tuning.
    """

    folder_names = get_augmented_folders(is_pre_training)

    func = lambda x: swap_files(x, is_pre_training)
    _ = list(map(func, folder_names))


def create_transfer_dataset(bigger_set, smaller_set):
    pre_training = bigger_set
    fine_tuning = smaller_set

    # Create directories
    os.makedirs(PRE_TRAIN_PATH)
    os.makedirs(FINE_TUNE_PATH)

    # Move folders
    move_folders(pre_training, PRE_TRAIN_PATH)
    move_folders(fine_tuning, FINE_TUNE_PATH)


def process_for_transfer_learning(bigger_set, smaller_set):
    # Get data for transfer learning
    create_transfer_dataset(bigger_set, smaller_set)

    # Split data for pre-training
    train_val_test_split(TRANSFER_TRAIN_RATIO, TRANSFER_VAL_RATIO, is_pre_training=True)
    solve_augmentation_issue(is_pre_training=True)

    # Split data for fine-tuning
    train_val_test_split(TRANSFER_TRAIN_RATIO, TRANSFER_VAL_RATIO, is_pre_training=False)
    solve_augmentation_issue(is_pre_training=False)

    
def main():
    character_folders = get_character_folders(DATA_PATH)
    character_folders = filter_character_folders(character_folders, 2)

    # Shuffle characters
    np.random.default_rng(0).shuffle(character_folders)

    bigger_set, smaller_set = split_character_folders(character_folders)
    
    # # Create data for meta-learning
    # create_meta_dataset(bigger_set, smaller_set)
    
    # Create data for transfer learning
    process_for_transfer_learning(bigger_set, smaller_set)


if __name__ == '__main__':
    main()
    