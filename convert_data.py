#############################################################################
# This file contains the code to convert the data                           #
# from the original format to the format used by the model.                 #
#                                                                           #
# A converted dataset for all literature works contains                     #
# the following folders/files:                                              #
#   - character<character_id>: contains all images of this character_id,    #
#   - meta_data.yaml: contains:                                             #
#       - n_classes: number of unique characters in this literature work.   #
#       - unicode_classes: a list of characters in form of unicode,         #
#                          the character_id of a class is its index number. #
#############################################################################

import os
import glob
import pandas as pd
import numpy as np
import cv2
import itertools


RAW_DATA_PATH = os.environ['_BASE_PATH']

PROC_PATH = os.environ['_PROC_PATH']

CHARACTER_FILE_EXTENSION = os.environ['_CHARACTER_FILE_EXTENSION']
META_DATA_FILE_NAME = os.environ['_META_DATA_FILE_NAME']

CHARACTER_WIDTH_SCALED = int(os.environ['_CHARACTER_WIDTH_SCALED'])
CHARACTER_HEIGHT_SCALED = int(os.environ['_CHARACTER_HEIGHT_SCALED'])

LUC_VAN_TIEN_IMAGES_PATH = os.path.join(RAW_DATA_PATH, "Luc Van Tien/lvt-raw-images")
LUC_VAN_TIEN_LABELS_PATH = os.path.join(RAW_DATA_PATH, "Luc Van Tien/lvt-mynom")

TALE_OF_KIEU_1866_IMAGES_PATH = os.path.join(RAW_DATA_PATH, "Tale of Kieu/1866-raw-images")
TALE_OF_KIEU_1866_LABELS_PATH = os.path.join(RAW_DATA_PATH, "Tale of Kieu/1866-mynom")

TALE_OF_KIEU_1871_IMAGES_PATH = os.path.join(RAW_DATA_PATH, "Tale of Kieu/1871-raw-images")
TALE_OF_KIEU_1871_LABELS_PATH = os.path.join(RAW_DATA_PATH, "Tale of Kieu/1871-mynom")

TALE_OF_KIEU_1902_IMAGES_PATH = os.path.join(RAW_DATA_PATH, "Tale of Kieu/1902-raw-images")
TALE_OF_KIEU_1902_LABELS_PATH = os.path.join(RAW_DATA_PATH, "Tale of Kieu/1902-mynom")

LEFT_COOR_NAME = "LEFT"
TOP_COOR_NAME = "TOP"
RIGHT_COOR_NAME = "RIGHT"
BOTTOM_COOR_NAME = "BOTTOM"

LABEL_NAME = "UNICODE"
UNKNOWN_CHARACTER = "UNK"

LABEL_NAME_TALE_OF_KIEU_1866 = "UNICODE1"
CONFIDENCE_NAME_TALE_OF_KIEU_1866 = "CONFIDENCE1"
CONFIDENCE_THRESHOLD = 0.9

RAW_IMAGE_FILE_EXTENSION = "jpg"
RAW_LABEL_FILE_EXTENSION = "xlsx"

FILE_NAME_COLUMN = "FILE_NAME"


def get_raw_path(file_extension, *argv):
    """ Gets all raw images/labels file paths from all literature works.
    
    Args:
        file_extension (str): desired file's extension
        *argv (str): list of paths
    Returns:
        raw_paths (list): list of all raw images/labels file paths
    """
    
    raw_paths = []
    for arg in argv:
        paths = glob.glob(
            os.path.join(arg, f'*.{file_extension}')
        )
        raw_paths.extend(paths)

    return raw_paths


def load_labels(label_paths):
    """ Loads labels from excel files

    Args:
        label_paths (list): list of paths to label files
    Returns:
        df_labels (DataFrame): dataframe containing labels
    """

    accepted_columns = [
        LEFT_COOR_NAME, 
        TOP_COOR_NAME, 
        RIGHT_COOR_NAME, 
        BOTTOM_COOR_NAME, 
        LABEL_NAME,
        FILE_NAME_COLUMN
    ]

    df_labels = pd.DataFrame()

    for label_path in label_paths:
        # Load label file
        df_img = pd.read_excel(label_path)

        # Add file name column
        df_img[FILE_NAME_COLUMN] = os.path.basename(label_path).split('.')[0]

        # Process for Tale of Kieu 1866 only
        if LABEL_NAME_TALE_OF_KIEU_1866 in df_img.columns:
            df_img = df_img[
                df_img[CONFIDENCE_NAME_TALE_OF_KIEU_1866] >= CONFIDENCE_THRESHOLD
            ]
            df_img.rename(
                columns = {LABEL_NAME_TALE_OF_KIEU_1866: LABEL_NAME}, 
                inplace = True
            )

        # Remove unused columns
        df_img = df_img[accepted_columns]

        df_labels = pd.concat([df_labels, df_img])

    df_labels.reset_index(inplace=True)
    df_labels.drop(columns=['index'], inplace=True)
    
    return df_labels


def unicode_to_nom(unicode):
    """ Convert unicode to Nôm text

    Args:
        unicode (str): unicode of a character
    """

    return chr(int(unicode, 16))


def remove_unknown_characters(df_labels):
    """ Removes unknown characters from labels.

    Args:
        df_labels (DataFrame): dataframe containing labels
    Returns:
        res (DataFrame): dataframe after removing unknown characters
    """

    # Remove unknown characters
    res = df_labels[df_labels[LABEL_NAME] != UNKNOWN_CHARACTER]

    return res


def remove_invalid_unicode(df_labels):
    """ Removes invalid unicode from labels.

    Args:
        df_labels (DataFrame): dataframe containing labels
    Returns:
        res (DataFrame): dataframe after removing invalid unicode
    """

    unicode_checker = lambda x: unicode_to_nom(x).isalpha()
    res = df_labels[df_labels[LABEL_NAME].apply(unicode_checker)]

    return res


def remove_invalid_characters(df_labels):
    """ Removes invalid characters from labels.
        The cases that a character is invalid:
            - The character is unknown.
            - The unicode of the character is invalid.

    Args:
        df_labels (DataFrame): dataframe containing labels
    Returns:
        res (DataFrame): dataframe after removing unknown characters
    """

    res = remove_unknown_characters(df_labels)
    res = remove_invalid_unicode(res)

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
    df_img = df_labels[df_labels[FILE_NAME_COLUMN] == file_name]
    
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
    func = lambda x: resize_character(x, (CHARACTER_WIDTH_SCALED, CHARACTER_HEIGHT_SCALED))
    character_images = list(map(func, character_images))
    
    # Get labels
    labels = df_img[LABEL_NAME]

    return tuple(zip(character_images, labels))


def get_characters(image_paths, df_labels):
    """ Gets all characters from all images.
    
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


def save_meta_data(unique_characters, output_file):
    """ Saves meta data to output file.
    
    Args:
        unique_characters (list): list of unique characters
        output_file (str): output file path
    """

    n_classes = len(unique_characters)

    with open(output_file, 'w') as f:
        # Write number of classes
        f.write("n_classes: " + str(n_classes) + '\n\n')

        # Write classes
        f.write("unicode_classes: [")
        for i, character in enumerate(unique_characters):
            if i == 0:
                f.write(character)
            else:
                f.write(", " + character)
        f.write("]")

        f.close()


def save_character(character, label, output_dir, character_list):
    """Saves a character to output directory.
    
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
    file_name = str(n_files_exist + 1).zfill(4) + "." + CHARACTER_FILE_EXTENSION
    file_path = os.path.join(label_folder_path, file_name)
    cv2.imwrite(file_path, character)


def save_characters(characters, output_dir, character_list):
    """ Saves characters to output directory.
    
    Args:
        characters (list): list of characters and their associated labels
        output_dir (str): output directory
        character_list (np.array): list of unique characters
    """
    
    func = lambda x: save_character(x[0], x[1], output_dir, character_list)
    _ = list(map(func, characters))


def main():
    image_paths = get_raw_path(
        RAW_IMAGE_FILE_EXTENSION,
        LUC_VAN_TIEN_IMAGES_PATH,
        TALE_OF_KIEU_1866_IMAGES_PATH,
        TALE_OF_KIEU_1871_IMAGES_PATH,
        TALE_OF_KIEU_1902_IMAGES_PATH
    )
    label_paths = get_raw_path(
        RAW_LABEL_FILE_EXTENSION,
        LUC_VAN_TIEN_LABELS_PATH,
        TALE_OF_KIEU_1866_LABELS_PATH,
        TALE_OF_KIEU_1871_LABELS_PATH,
        TALE_OF_KIEU_1902_LABELS_PATH
    )

    df_labels = load_labels(label_paths)
    df_labels = remove_invalid_characters(df_labels)

    characters = get_characters(image_paths, df_labels)

    unique_characters = np.unique(df_labels[LABEL_NAME])
    unique_characters = np.sort(unique_characters)

    meta_file = os.path.join(PROC_PATH, META_DATA_FILE_NAME)
    save_meta_data(unique_characters, meta_file)

    save_characters(characters, PROC_PATH, unique_characters)


if __name__ == "__main__":
    main()
