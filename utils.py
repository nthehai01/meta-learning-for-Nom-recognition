import cv2
import torch
import os


OPTIMIZER_STATE_NAME="optimizer"
PARAMETERS_STATE_NAME="parameters"


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


def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()


def save_checkpoint(optimizer, parameters, checkpoint_step, log_dir):
    """ Saves a training checkpoint.

    Args:
        optimizer (torch.optim): optimizer used for training
        parameters (dict): model parameters
        checkpoint_step (int): training step
        log_dir (str): directory to save checkpoint
    """


    checkpoint = {
        OPTIMIZER_STATE_NAME: optimizer.state_dict(),
        PARAMETERS_STATE_NAME: parameters
    }
    
    checkpoint_file = os.path.join(log_dir, f'state_{checkpoint_step}.pt')
    torch.save(checkpoint, checkpoint_file)

    print('Saved checkpoint.')


def load_checkpoint(optimizer, parameters, checkpoint_step, log_dir, device="cpu"):
    """ Loads a training checkpoint.

    Args:
        optimizer (torch.optim): optimizer used for training
        parameters (dict): model parameters
        checkpoint_step (int): training step
        log_dir (str): directory of saved checkpoint
        device (str): device to load checkpoint to
    Raises:
        ValueError: if checkpoint for checkpoint_step is not found    
    """

    checkpoint_file = os.path.join(log_dir, f'state_{checkpoint_step}.pt')

    if os.path.isfile(checkpoint_file):
        state = torch.load(checkpoint_file, map_location=device)
        optimizer.load_state_dict(state[OPTIMIZER_STATE_NAME])
        parameters.update(state[PARAMETERS_STATE_NAME])

        print(f'Loaded checkpoint iteration {checkpoint_step}.')
    else:
        raise ValueError(
            f'No checkpoint for iteration {checkpoint_step} found.'
        )
    

def tensorboard_writer(writer, tag_value_pairs, step):
    """ Writes scalar values to tensorboard.

    Args:
        writer (tensorboardX.SummaryWriter): tensorboard writer
        tag_value_pairs (dict): dictionary of tag-value pairs
        step (int): training step
    """

    for tag, value in tag_value_pairs.items():
        writer.add_scalar(tag, value, step)
    