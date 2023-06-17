#################################################
# Run file for experimenting transfer-learning. #
#################################################

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from torch.utils import tensorboard
import numpy as np

from transfer_learning.model import NomModel
from transfer_learning.nom_dataset import get_nom_dataloader


BIGGER_SET_CHARACTER_NUM = int(os.environ["_BIGGER_SET_CHARACTER_NUM"])
SMALLER_SET_CHARACTER_NUM = int(os.environ["_SMALLER_SET_CHARACTER_NUM"])
PRE_TRAIN_NAME = os.environ["_PRE_TRAIN_NAME"]
FINE_TUNE_NAME = os.environ["_FINE_TUNE_NAME"]


def parse_arguments():
    """ Parses the command line arguments of the program."""

    parser = argparse.ArgumentParser("Experiment Transfer-learning.")
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='Directory to save or load logs'
    )

    parser.add_argument(
        '--finetune', 
        default=False, 
        action='store_true',
        help='Whether to pretrain or finetune the model'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of tasks per batch'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--num_train_iterations',
        type=int,
        default=10001,
        help='Number of updates to train for'
    )

    parser.add_argument(
        '--test',
        default=False,
        action='store_true',
        help='Whether to test or train the model'
    )

    parser.add_argument(
        '--checkpoint_step',
        type=int,
        default=-1,
        help='Checkpoint iteration to load for resuming training/testing (-1 to ignored)'
    )

    # For fine-tuning only
    parser.add_argument(
        '--pretrained_weights',
        type=str,
        default=None,
        help='[For fine-tuning only] Pre-trained weights initialed for fine-tuning (-1 to ignored)'
    )

    return parser.parse_args()


def main():
    """ Main function of the program."""

    # Argument parsing
    args = parse_arguments()

    phase = FINE_TUNE_NAME if args.finetune else PRE_TRAIN_NAME

    # Tensorboard writer
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/{phase}/nom_transfer.batch_size:{args.batch_size}.lr:{args.lr}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    # Model
    output_dim = SMALLER_SET_CHARACTER_NUM if args.finetune else BIGGER_SET_CHARACTER_NUM
    net = NomModel(output_dim, args.lr, log_dir)
    
    if args.checkpoint_step > -1:  # Load checkpoint if needed
        net.load_checkpoint(args.checkpoint_step)
    elif args.finetune and args.pretrained_weights:  # Load pre-trained weights and initialize fine-tuning
        net.load_pre_trained_weights(args.pretrained_weights)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:  # Train network
        num_training_images = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        
        dataloader_train = get_nom_dataloader(
            phase,
            'train',
            args.batch_size,
            num_training_images
        )
        dataloader_val = get_nom_dataloader(
            phase,
            'train',
            args.batch_size,
            args.batch_size * 10
        )

        net.train(dataloader_train, dataloader_val, writer)
    else:  # Test network
        print("Testing...")

        dataloader_test = get_nom_dataloader(
            phase,
            'test',
            1,
            600*4
        )

        # Perform testing
        _, accuracies = net.test(dataloader_test)

        # Compute statistics
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(len(accuracies))
        print(
            f'Accuracy over {len(accuracies)} test images: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )


if __name__ == '__main__':
    main()
