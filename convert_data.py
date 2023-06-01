#############################################################
# This file contains the code to convert the data           #
# from the original format to the format used by the model. #
#############################################################

import os
import glob
import pandas as pd
import numpy as np
import cv2
import itertools


LUC_VAN_TIEN_IMAGES_PATH = os.environ['_LUC_VAN_TIEN_IMAGES_PATH']
LUC_VAN_TIEN_LABELS_PATH = os.environ['_LUC_VAN_TIEN_LABELS_PATH']
PROC_LUC_VAN_TIEN_PATH = os.environ['_PROC_LUC_VAN_TIEN_PATH']

TALE_OF_KIEU_1866_IMAGES_PATH = os.environ['_TALE_OF_KIEU_1866_IMAGES_PATH']
TALE_OF_KIEU_1866_LABELS_PATH = os.environ['_TALE_OF_KIEU_1866_LABELS_PATH']
PROC_TALE_OF_KIEU_1866_PATH = os.environ['_PROC_TALE_OF_KIEU_1866_PATH']

TALE_OF_KIEU_1871_IMAGES_PATH = os.environ['_TALE_OF_KIEU_1871_IMAGES_PATH']
TALE_OF_KIEU_1871_LABELS_PATH = os.environ['_TALE_OF_KIEU_1871_LABELS_PATH']
PROC_TALE_OF_KIEU_1871_PATH = os.environ['_PROC_TALE_OF_KIEU_1871_PATH']

TALE_OF_KIEU_1902_IMAGES_PATH = os.environ['_TALE_OF_KIEU_1902_IMAGES_PATH']
TALE_OF_KIEU_1902_LABELS_PATH = os.environ['_TALE_OF_KIEU_1902_LABELS_PATH']
PROC_TALE_OF_KIEU_1902_PATH = os.environ['_PROC_TALE_OF_KIEU_1902_PATH']

LEFT_COOR_NAME = os.environ['_LEFT_COOR_NAME']
TOP_COOR_NAME = os.environ['_TOP_COOR_NAME']
RIGHT_COOR_NAME = os.environ['_RIGHT_COOR_NAME']
BOTTOM_COOR_NAME = os.environ['_BOTTOM_COOR_NAME']

LABEL_NAME = os.environ['_LABEL_NAME']
UNKNOWN_CHARACTER = os.environ['_UNKNOWN_CHARACTER']

LABEL_NAME_TALE_OF_KIEU_1866 = os.environ['_LABEL_NAME_TALE_OF_KIEU_1866']
CONFIDENCE_NAME_TALE_OF_KIEU_1866 = os.environ['_CONFIDENCE_NAME_TALE_OF_KIEU_1866']
CONFIDENCE_THRESHOLD = float(os.environ['_CONFIDENCE_THRESHOLD'])

WIDTH_SCALED = int(os.environ['_WIDTH_SCALED'])
HEIGHT_SCALED = int(os.environ['_HEIGHT_SCALED'])


def load_labels(label_paths):
    """ Loads labels from excel files

    Args:
        label_paths (list): list of paths to label files
    Returns:
        df_labels (DataFrame): dataframe containing images and labels
    """

    df_labels = pd.DataFrame()

    for label_path in label_paths:
        df_img = pd.read_excel(label_path)
        df_img['FILE_NAME'] = os.path.basename(label_path).split('.')[0]
        df_labels = pd.concat([df_labels, df_img])

    df_labels.reset_index(inplace=True)
    df_labels.drop(columns=['index'], inplace=True)

    # Process for Tale of Kieu 1866
    if LABEL_NAME_TALE_OF_KIEU_1866 in df_labels.columns:
        df_labels = df_labels[
            df_labels[CONFIDENCE_NAME_TALE_OF_KIEU_1866] >= CONFIDENCE_THRESHOLD
        ]
        df_labels.rename(
            columns = {LABEL_NAME_TALE_OF_KIEU_1866: LABEL_NAME}, 
            inplace = True
        )
    
    return df_labels


def remove_unknown_characters(df_labels):
    """ Removes unknown characters from labels.

    Args:
        df_labels (DataFrame): dataframe containing images and labels
    Returns:
        res (DataFrame): dataframe after removing unknown characters
    """

    res = df_labels[df_labels[LABEL_NAME] != UNKNOWN_CHARACTER]
    return res


def load_image(file_path):
    """ Loads an image from a file path.

    Args:
        file_path (str): file path of image
    Returns:
        a Tensor containing BGR image 
            of arbitrary shape (height, width, 3)
    """

    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return image


def split_image(image, left, top, right, bottom):
    """ Splits an image into a character image.

    Args:
        image (Tensor): image data
        left (int): left coordinate of bounding box
        top (int): top coordinate of bounding box
        right (int): right coordinate of bounding box
        bottom (int): bottom coordinate of bounding box
    Returns:
        a Tensor containing character image 
            of shape (bottom-top, right-left, 3)
    """

    crop = image[top:bottom, left:right]
    return crop


def resize_character(character, size):
    """ Resizes a character image to a fixed size.
    
    Args:
        character (Tensor): character image
        size (tuple): desired size
    Returns:
        a Tensor containing resized character image
    """

    resized = cv2.resize(character, size, interpolation=cv2.INTER_LINEAR)
    return resized


def get_characters_from_image(image_path, df_labels):
    """ Gets the character images and their associated labels from an image.
    
    Args:
        image_path (str): file path of image
        df_labels (DataFrame): DataFrame containing labels
    Returns:
        a 2-value tuple containing character images and their associated labels
    """

    # Load image
    image = load_image(image_path)
    
    # Get character bounding box information for this image
    file_name = os.path.basename(image_path).split('.')[0]
    df_img = df_labels[df_labels['FILE_NAME'] == file_name]
    
    # Split image into character images
    func = lambda x: split_image(
        image, 
        x[LEFT_COOR_NAME], 
        x[TOP_COOR_NAME], 
        x[RIGHT_COOR_NAME], 
        x[BOTTOM_COOR_NAME]
    )
    character_images = df_img.apply(func, axis=1)

    # Resize character images
    func = lambda x: resize_character(x, (WIDTH_SCALED, HEIGHT_SCALED))
    character_images = list(map(func, character_images))
    
    # Get labels
    labels = df_img[LABEL_NAME]

    return tuple(zip(character_images, labels))


def get_characters(image_paths, df_labels):
    """ Get all characters from all images.
    
    Args:
        image_paths (list): list of image file paths
        df_labels (DataFrame): DataFrame containing labels
    Returns:
        a list of characters and their associated labels
    """
    
    # Get characters from each image
    func = lambda path: get_characters_from_image(path, df_labels)
    characters = list(map(func, image_paths))

    # Flatten the list
    characters = list(itertools.chain(*characters))

    return characters


def save_character(character, label, output_dir, character_list):
    """Save a character to output directory.
    
    Args:
        character (Tensor): character image
        label (str): character label
        output_dir (str): output directory
        character_list (np.array): list of unique characters
    """

    # Create label folder
    # label_index = character_list.index(label)
    label_index = np.where(character_list == label)[0][0]
    label_folder_name = "character" + str(label_index).zfill(5)
    label_folder_path = os.path.join(output_dir, label_folder_name)
    os.makedirs(label_folder_path, exist_ok=True)

    # Save character image
    n_files_exist = len(os.listdir(label_folder_path))
    file_name = str(n_files_exist + 1).zfill(4) + ".png"
    file_path = os.path.join(label_folder_path, file_name)
    cv2.imwrite(file_path, character)


def save_characters(characters, output_dir, character_list):
    """ Save characters to output directory.
    
    Args:
        characters (list): list of characters and their associated labels
        output_dir (str): output directory
        character_list (np.array): list of unique characters
    """
    
    func = lambda x: save_character(x[0], x[1], output_dir, character_list)
    _ = list(map(func, characters))


def save_meta_data(unique_characters, output_file):
    """ Save meta data to output file.
    
    Args:
        unique_characters (list): list of unique characters
        output_file (str): output file path
    """

    n_classes = len(unique_characters)

    with open(output_file, 'w') as f:
        # Write number of classes
        f.write("n_classes: " + str(n_classes) + '\n\n')

        # Write classes
        f.write("classes: [")
        for i, character in enumerate(unique_characters):
            if i == 0:
                f.write("'" + character + "'")
            else:
                f.write(", '" + character + "'")
        f.write("]")

        f.close()


def process_data(image_paths, label_paths, output_dir):
    """ For a specific literature work, create a folder containing all characters.
    
    Args:
        image_paths (list): list of image file paths
        label_paths (list): list of label file paths
        output_dir (str): output directory
    """
    
    df_labels = load_labels(label_paths)

    df_labels = remove_unknown_characters(df_labels)
    
    characters = get_characters(image_paths, df_labels)
    
    unique_characters = np.unique(df_labels[LABEL_NAME])
    unique_characters = np.sort(unique_characters)
    
    meta_file = os.path.join(output_dir, "meta.txt")
    save_meta_data(unique_characters, meta_file)

    save_characters(characters, output_dir, unique_characters)


if __name__ == '__main__':
    # Luc Van Tien
    print("Processing Luc Van Tien...")
    image_paths = glob.glob(os.path.join(LUC_VAN_TIEN_IMAGES_PATH, '*.jpg'))
    label_paths = glob.glob(os.path.join(LUC_VAN_TIEN_LABELS_PATH, '*.xlsx'))
    process_data(image_paths, label_paths, PROC_LUC_VAN_TIEN_PATH)

    # Tale of Kieu 1866
    print("Processing Tale of Kieu 1866...")
    image_paths = glob.glob(os.path.join(TALE_OF_KIEU_1866_IMAGES_PATH, '*.jpg'))
    label_paths = glob.glob(os.path.join(TALE_OF_KIEU_1866_LABELS_PATH, '*.xlsx'))
    process_data(image_paths, label_paths, PROC_TALE_OF_KIEU_1866_PATH)

    # Tale of Kieu 1871
    print("Processing Tale of Kieu 1871...")
    image_paths = glob.glob(os.path.join(TALE_OF_KIEU_1871_IMAGES_PATH, '*.jpg'))
    label_paths = glob.glob(os.path.join(TALE_OF_KIEU_1871_LABELS_PATH, '*.xlsx'))
    process_data(image_paths, label_paths, PROC_TALE_OF_KIEU_1871_PATH)

    # Tale of Kieu 1902
    print("Processing Tale of Kieu 1902...")
    image_paths = glob.glob(os.path.join(TALE_OF_KIEU_1902_IMAGES_PATH, '*.jpg'))
    label_paths = glob.glob(os.path.join(TALE_OF_KIEU_1902_LABELS_PATH, '*.xlsx'))
    process_data(image_paths, label_paths, PROC_TALE_OF_KIEU_1902_PATH)
