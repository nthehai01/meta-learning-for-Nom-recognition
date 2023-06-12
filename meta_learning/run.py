########################################################
# Run file for experimenting meta-learning algorithms: #
#   1. ProtoNet                                        #
#   2. MAML                                            #
#   3. ProtoMAML                                       #  
########################################################

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from torch.utils import tensorboard
import numpy as np

from meta_learning.methods.protonet import ProtoNet
from meta_learning.methods.maml import MAML
from meta_learning.methods.proto_maml import ProtoMAML
from meta_learning.nom_dataset import get_nom_dataloader


NUM_TEST_TASKS = int(os.environ["_NUM_TEST_TASKS"])


def parse_arguments():
    """ Parses the command line arguments of the program.
    """

    parser = argparse.ArgumentParser("Experiment some Meta-learning algorithms.")
    parser.add_argument(
        '--method',
        type=str,
        default=None,
        choices=["protonet", "maml", "protomaml"],
        help="One of Meta-learning algorithms [protonet, maml, protomaml]"
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='Directory to save or load logs'
    )
    parser.add_argument(
        '--num_way',
        type=int,
        default=5,
        help='Number of classes in a task'
    )
    parser.add_argument(
        '--num_shot',
        type=int,
        default=1,
        help='Number of support examples per class in a task'
    )
    parser.add_argument(
        '--num_query',
        type=int,
        default=5,
        help='Maximum number of query examples per class in a task'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of tasks per batch'
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

    # For ProtoNet only
    parser.add_argument(
        '--output_dim',
        type=int,
        default=64,
        help='[For ProtoNet only] Dimensionality of output space'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='[For ProtoNet only] Learning rate'
    )

    # For MAML and ProtoMAML only
    parser.add_argument(
        '--num_inner_steps',
        type=int,
        default=1,
        help='[For MAML and ProtoMAML only] Number of inner loop updates'
    )
    parser.add_argument(
        '--inner_lr',
        type=float,
        default=0.4,
        help='[For MAML and ProtoMAML only] Inner loop learning rate'
    )
    parser.add_argument(
        '--outer_lr',
        type=float,
        default=0.001,
        help='[For MAML and ProtoMAML only] Outer loop learning rate'
    )

    return parser.parse_args()


def determine_algorithm(args, log_dir):
    """ Determines and instantiates the algorithm to use based on the method name.

    Args:
        args: Arguments of the program, contains method name 
            as well as some vital arguments to instantiating a network.
        log_dir (str): Directory to save or load logs.
    Returns:
        net: Network instance to use.
    Raises:
        ValueError: If the method name is invalid.
    """

    if args.method == 'protonet':
        net = ProtoNet(
            args.output_dim,
            args.lr,
            log_dir
        )
    elif args.method == 'maml':
        net = MAML(
            args.num_way, 
            args.num_inner_steps, 
            args.inner_lr, 
            args.outer_lr, 
            log_dir
        )
    elif args.method == 'protomaml':
        net = ProtoMAML(
            args.output_dim, 
            args.num_inner_steps, 
            args.inner_lr, 
            args.outer_lr, 
            log_dir
        )
    else:
        raise ValueError(f'Invalid method name: {args.method}')

    return net


def main():
    """ Main function of the program.
    """

    # Argument parsing
    args = parse_arguments()

    # Tensorboard writer
    log_dir = args.log_dir
    if log_dir is None:
        if args.method == 'protonet':
            log_dir = f'./logs/{args.method}/nom_meta.way:{args.num_way}.shot:{args.num_shot}.query:{args.num_query}.batch_size:{args.batch_size}.lr:{args.lr}'
        else:  # maml or protomaml
            log_dir = f'./logs/{args.method}/nom_meta.way:{args.num_way}.shot:{args.num_shot}.query:{args.num_query}.batch_size:{args.batch_size}.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.outer_lr:{args.outer_lr}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    # Determine meta algorithm to use
    net = determine_algorithm(args, log_dir)

    # Load checkpoint if needed
    if args.checkpoint_step > -1:
        net.load_checkpoint(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:  # Train network
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_shot={args.num_shot}, '
            f'max_num_query={args.num_query}'
        )

        dataloader_train = get_nom_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_shot,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = get_nom_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_shot,
            args.num_query,
            args.batch_size * 4
        )

        net.train(dataloader_train, dataloader_val, writer)
    else:  # Test network
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_shot={args.num_shot}, '
            f'max_num_query={args.num_query}'
        )

        dataloader_test = get_nom_dataloader(
            'test',
            1,
            args.num_way,
            args.num_shot,
            args.num_query,
            NUM_TEST_TASKS
        )

        # Perform testing
        _, _, accuracies_query = net.test(dataloader_test)

        # Compute statistics
        mean = np.mean(accuracies_query)
        std = np.std(accuracies_query)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )


if __name__ == '__main__':
    main()
