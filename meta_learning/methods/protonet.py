################################################################
# Implementation of Prototypical Networks for Nôm recognition. #
################################################################

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import torch
import torch.nn.functional as F

from utils import score, save_checkpoint, load_checkpoint, tensorboard_writer
from backbone import Backbone


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = int(os.environ["_SAVE_INTERVAL"])
PRINT_INTERVAL = int(os.environ["_PRINT_INTERVAL"])
VAL_INTERVAL = int(os.environ["_VAL_INTERVAL"])
NUM_TEST_TASKS = int(os.environ["_NUM_TEST_TASKS"])


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, output_dim, learning_rate, log_dir):
        """ Initializes a prototypical network.

        Args:
            output_dim (int): dimensionality of output space
            learning_rate (float): learning rate for training
            log_dir (str): directory to save model checkpoints
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
                shape (n, feat_dim)
        """

        return Backbone.forward(images, self._network)


    @staticmethod
    def compute_prototypes(features, labels):
        """ Computes class prototypes from support features.

        Args:
            features (torch.Tensor): support features
                shape (num_way * num_shot, feat_dim)
            labels (torch.Tensor): support labels
                shape (num_way * num_shot)
        Returns:
            prototypes (torch.Tensor): class prototypes as a Tensor
                shape (num_way, feat_dim)
        """

        classes, _ = torch.unique(labels).sort()

        prototypes = []
        for c in classes:
            features_class = features[labels == c]
            prototype = torch.mean(features_class, dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)

        return prototypes


    def _compute_distances(self, features, prototypes):
        """ Computes squared distances between query features and prototypes.
        
        Args:
            features (torch.Tensor): query features
                shape (n, feat_dim)
            prototypes (torch.Tensor): class prototypes
                shape (num_way, feat_dim)
        Returns:
            distances (torch.Tensor): distances between features and prototypes
                shape (n, num_way)
        """

        diff = prototypes[None, :] - features[:, None]
        distances = torch.pow(diff, 2).sum(dim=2)
        
        return distances
    

    def _classify_feats(self, images, labels, prototypes):
        """ Classifies features based on their distances to prototypes.
        
        Args:
            images (torch.Tensor): images to classify
                shape (n, feat_dim)
            labels (torch.Tensor): query labels
                shape (n,)
            prototypes (torch.Tensor): class prototypes
                shape (num_way, feat_dim)
        Returns:
            losses (torch.Tensor): mean cross-entropy loss over query set
                shape ()
            accuracies (torch.Tensor): classification accuracy on query set
                shape ()
        """

        # Compute the distance from each example to each prototype
        features = self._forward(images)
        distances = self._compute_distances(features, prototypes)

        # Make predictions
        preds = F.log_softmax(-distances, dim=1)

        # Compute classification losses
        losses = F.cross_entropy(preds, labels)

        # Compute accuracies
        accuracies = score(preds, labels)

        return losses, accuracies
    

    def _step(self, task_batch):
        """ Performs a single training step.
            
            For a given task in the task_batch, compute the prototypes, 
            classification accuracies on support and query sets, 
            and the loss on query set.
        
        Args:
            task_batch (tuple): a batch of tasks
        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """

        loss_batch, accuracy_support_batch, accuracy_query_batch = [], [], []

        for task in task_batch:
            # Get task data
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)

            # Compute prototypes
            features_support = self._forward(images_support)
            prototypes = self.compute_prototypes(features_support, labels_support)

            # Compute accuracy on support set
            _, accuracies_support = self._classify_feats(
                images_support, 
                labels_support, 
                prototypes
            )

            # Compute loss and accuracy on query set
            losses, accuracies_query = self._classify_feats(
                images_query, 
                labels_query, 
                prototypes
            )

            # Log metrics
            loss_batch.append(losses)
            accuracy_support_batch.append(accuracies_support)
            accuracy_query_batch.append(accuracies_query)

        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch).item(),
            np.mean(accuracy_query_batch).item()
        )


    def train(self, train_loader, val_loader, writer):
        """ Trains the prototypical network.

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
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            loss = loss.item()

            # Log training metrics
            if train_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {train_step}: '
                    f'loss: {loss:.3f}, '
                    f'accuracy support: {accuracy_support:.3f}, '
                    f'accuracy query: {accuracy_query:.3f}'
                )

                metric_pairs = {
                    'loss/train': loss,
                    'accuracy_support/train': accuracy_support,
                    'accuracy_query/train': accuracy_query
                }
                tensorboard_writer(writer, metric_pairs, train_step)

            # Evaluate on validation tasks
            if train_step % VAL_INTERVAL == 0:
                losses, accuracies_support, accuracies_query = self.test(val_loader)

                loss = np.mean(losses)
                accuracy_support = np.mean(accuracies_support)
                accuracy_query = np.mean(accuracies_query)

                metric_pairs = {
                    'loss/val': loss,
                    'accuracy_support/val': accuracy_support,
                    'accuracy_query/val': accuracy_query
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
        """ Tests the prototypical network.
        
        Args:
            test_loader (torch.utils.data.DataLoader): test data
        Returns:
            losses (list): losses for each task
            accuracies_support (list): accuracies on support set for each task
            accuracies_query (list): accuracies on query set for each task
        """

        losses, accuracies_support, accuracies_query = [], [], []

        with torch.no_grad():
            for task_batch in test_loader:
                loss, accuracy_support, accuracy_query = self._step(task_batch)
                
                losses.append(loss.item())
                accuracies_support.append(accuracy_support)
                accuracies_query.append(accuracy_query)

        return losses, accuracies_support, accuracies_query
    

    def load_checkpoint(self, checkpoint_step):
        """ Loads a checkpoint.
        
        Args:
            checkpoint_step (int): checkpoint iteration
        """

        load_checkpoint(
            self._optimizer, 
            self._network, 
            checkpoint_step, 
            self._log_dir,
            DEVICE
        )
        self._start_train_step = checkpoint_step + 1
