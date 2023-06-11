import cv2
import torch


def load_image(file_path):
    """Loads and transforms an Nôm character image.

    Args:
        file_path (str): file path of image
    Returns:
        a Tensor containing grayscale image data
            shape (1, h, w)
    """

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # shape of (h, w)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.reshape(1, *img.shape) # shape of (1, h, w)
    img = img / 255.0
    return img


def identity(x):
    """Identity function.

    Args:
        x (any): input
    Returns:
        x (any): input
    """
    return x


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
