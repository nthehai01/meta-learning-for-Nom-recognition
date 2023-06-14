######################################################
# Implementation of Nôm model for transfer learning. #
######################################################

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy as np
import torch.nn.functional as F


from utils import score, save_checkpoint, load_checkpoint, tensorboard_writer, load_pre_trained_weights
from backbone import Backbone


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = int(os.environ["_SAVE_INTERVAL"])
PRINT_INTERVAL = int(os.environ["_PRINT_INTERVAL"])
VAL_INTERVAL = int(os.environ["_VAL_INTERVAL"])
NUM_TEST_TASKS = int(os.environ["_NUM_TEST_TASKS"])


class NomModel(torch.nn.Module):
    """Nôm model for transfer learning."""

    def __init__(self, output_dim, learning_rate, log_dir):
        """ Initialize the model.

        Args:
            output_dim (int): output dimension of the model
            learning_rate (float): learning rate of the model
            log_dir (str): directory to save or load model checkpoints
        """

        self._network = Backbone.get_network(num_outputs=output_dim, device=DEVICE)

        self._optimizer = torch.optim.Adam(
            self._network.values(),
            lr=learning_rate
        )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0


    def _forward(self, images):
        """ Computes the forward pass.

        Args:
            images (torch.Tensor): images to classify
                shape (n, c, h, w)
        Returns:
            A Tensor consisting of logits for each class
                shape (n, output_dim)
        """

        return Backbone.forward(images, self._network)
    

    def _get_images_labels(self, batch):
        """ Gets images and labels from batch.

        Args:
            batch (list of tuple[tensor, int]): batch of data
        Returns:
            images (torch.Tensor): images to classify
                shape (n, c, h, w)
            labels (torch.Tensor): labels of images
                shape (n)
        """

        images, labels = list(zip(*batch))

        images = torch.stack(images).to(DEVICE)
        labels = torch.tensor(labels).to(DEVICE)

        return images, labels
    

    def _classify_feats(self, images, labels):
        """ Classifies features.

        Args:
            images (torch.Tensor): images to classify
                shape (n, c, h, w)
            labels (torch.Tensor): query labels
                shape (n,)
        Returns:
            loss (torch.Tensor): mean cross-entropy loss over query set
                shape ()
            accuracy (torch.Tensor): mean classification accuracy on query set
                shape ()
        """

        # Make predictions
        features = self._forward(images)
        preds = F.log_softmax(features, dim=1)

        # Compute classification loss
        loss = F.cross_entropy(preds, labels)

        # Compute accuracy
        accuracy = score(preds, labels)

        return loss, accuracy


    def _step(self, batch):
        images, labels = self._get_images_labels(batch)

        loss_batch, accuracy_batch = self._classify_feats(images, labels)

        return loss_batch, accuracy_batch
    

    def train(self, train_loader, val_loader, writer):
        """ Trains the model.

        Args:
            train_loader (torch.utils.data.DataLoader): training data
            val_loader (torch.utils.data.DataLoader): validation data
            writer (tensorboardX.SummaryWriter): tensorboard writer
        """

        print(f'Starting training at iteration {self._start_train_step}...')

        for train_step, task_batch in enumerate(train_loader, 
                                                start=self._start_train_step):
            # Train on task batch
            self._optimizer.zero_grad()
            loss, accuracy = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            loss = loss.item()

            # Log training metrics
            if train_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {train_step}: '
                    f'loss: {loss:.3f}, '
                    f'accuracy: {accuracy:.3f}, '
                )

                metric_pairs = {
                    'loss/train': loss,
                    'accuracy/train': accuracy,
                }
                tensorboard_writer(writer, metric_pairs, train_step)

            # Evaluate on validation tasks
            if train_step % VAL_INTERVAL == 0:
                losses, accuracies = self.test(val_loader)

                loss = np.mean(losses)
                accuracy = np.mean(accuracies)

                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'accuracy: {accuracy:.3f}, '
                )

                metric_pairs = {
                    'loss/val': loss,
                    'accuracy/val': accuracy,
                }
                tensorboard_writer(writer, metric_pairs, train_step)

            # Save model
            if train_step % SAVE_INTERVAL == 0:
                save_checkpoint(
                    self._optimizer,
                    self._network,
                    train_step,
                    self._log_dir
                )


    def test(self, test_loader):
        """ Tests the algorithm.
        
        Args:
            test_loader (torch.utils.data.DataLoader): test data
        Returns:
            losses (list): losses for each batch
            accuracies (list): accuracies on each batch
        """

        losses, accuracies = [], []

        with torch.no_grad():
            for task_batch in test_loader:
                loss, accuracy = self._step(task_batch)
                
                losses.append(loss.item())
                accuracies.append(accuracy)

        return losses, accuracies
    

    def load_checkpoint(self, checkpoint_step):
        """ Loads a checkpoint.
        
        Args:
            checkpoint_step (int): checkpoint iteration
            is_test (bool): whether to load the model for testing
        """

        load_checkpoint(
            self._optimizer, 
            self._network, 
            checkpoint_step, 
            self._log_dir,
            DEVICE,
        )

        self._start_train_step = checkpoint_step + 1

    
    def load_pre_trained_weights(self, pre_trained_weights_path):
        """ Loads pre-trained weights and initialize the model for fine-tuning.

        Args:
            pre_trained_weights_path (str): path to pre-trained weights
        """

        load_pre_trained_weights(self._network, pre_trained_weights_path, DEVICE)
