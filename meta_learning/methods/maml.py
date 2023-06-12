#######################################################################
# Implementation of Model-Agnostic Meta-Learning for Nôm recognition. #
#######################################################################


import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy as np
import torch.nn.functional as F


from utils import score, save_checkpoint, load_checkpoint, tensorboard_writer
from backbone import Backbone


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = int(os.environ["_SAVE_INTERVAL"])
PRINT_INTERVAL = int(os.environ["_PRINT_INTERVAL"])
VAL_INTERVAL = int(os.environ["_VAL_INTERVAL"])
NUM_TEST_TASKS = int(os.environ["_NUM_TEST_TASKS"])


class MAML:
    """Trains and assesses MAML."""

    def __init__(self, 
                 output_dim, 
                 num_inner_steps, 
                 inner_lr, 
                 outer_lr, 
                 log_dir):
        """ Initializes MAML.

        Args: 
            output_dim (int): dimensionality of output space, 
                i.e. num_way
            num_inner_steps (int): number of inner steps
            inner_lr (float): inner learning rate
            outer_lr (float): outer learning rate
            log_dir (str): directory to save or load model checkpoints
        """

        self._network = Backbone.get_network(num_outputs=output_dim, device=DEVICE)

        self._num_inner_steps = num_inner_steps
        self._inner_lr = inner_lr

        self._optimizer = torch.optim.Adam(
            self._network.values(),
            lr=outer_lr
        )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0


    def _forward(self, images, parameters):
        """ Computes the forward pass.

        Args:
            images (torch.Tensor): images to classify
                shape (n, c, h, w)
            parameters (dict[str, Tensor]): parameters to use for
                the computation
        Returns:
            A Tensor consisting of logits for each class
                shape (n, num_way)
        """

        return Backbone.forward(images, parameters)
    

    def _classify_feats(self, images, labels, parameters):
        """ Classifies features.

        Args:
            images (torch.Tensor): images to classify
                shape (n, c, h, w)
            labels (torch.Tensor): query labels
                shape (n,)
            parameters (dict[str, Tensor]): parameters to use for
                the computation
        Returns:
            loss (torch.Tensor): mean cross-entropy loss over query set
                shape ()
            accuracy (torch.Tensor): mean classification accuracy on query set
                shape ()
        """

        # Make predictions
        features = self._forward(images, parameters)
        preds = F.log_softmax(features, dim=1)

        # Compute classification loss
        loss = F.cross_entropy(preds, labels)

        # Compute accuracy
        accuracy = score(preds, labels)

        return loss, accuracy
    

    def _init_local_parameters(self):
        """ Initializes task-specific parameters by cloning the model's meta-params.

        Returns:
            local_parameters (dict[str, Tensor]): local parameters
        """

        return {
            k: torch.clone(v)
            for k, v in self._network.items()
        }
    

    def _adapt(self, images_support, labels_support, local_parameters):
        """ Performs adaptations in the inner loop.

        Args:
            images_support (torch.Tensor): support images
                shape (num_way * num_shot, c, h, w)
            labels_support (torch.Tensor): support labels
                shape (num_way * num_shot,)
            local_parameters (dict[str, Tensor]): pre-adapted local parameters
        Returns:
            local_parameters (dict[str, Tensor]): adapted parameters
            accuracies_support (np.array): support set accuracy over the
                course of the inner loop
                shape (num_inner_steps + 1,)
        """
        
        accuracies_support = []

        # Perform adaptation
        for _ in range(self._num_inner_steps):
            # Compute accuracy on support set during adaptation
            loss_support, accuracy_support = self._classify_feats(
                images_support, 
                labels_support, 
                local_parameters
            )
            accuracies_support.append(accuracy_support)

            # Compute gradients
            grads = torch.autograd.grad(loss_support, local_parameters.values())

            # Update parameters
            local_parameters = {
                k: v - self._inner_lr * g
                for k, v, g in zip(local_parameters.keys(), local_parameters.values(), grads)
            }

        # Compute accuracy on support set after adaptation
        _, accuracy_support = self._classify_feats(
            images_support, 
            labels_support, 
            local_parameters
        )
        accuracies_support.append(accuracy_support)

        return local_parameters, accuracies_support


    def _inner_loop(self, images_support, labels_support):
        """ Performs an inner loop.

        Args:
            images_support (torch.Tensor): support images
                shape (num_way * num_shot, c, h, w)
            labels_support (torch.Tensor): support labels
                shape (num_way * num_shot,)
        Returns:
            local_parameters (dict[str, Tensor]): adapted parameters
            accuracies_support (np.array): support set accuracy over the
                course of the inner loop
                shape (num_inner_steps + 1,)
        """

        # Initialize local parameters
        local_parameters = self._init_local_parameters()

        # Perform adaptation
        local_parameters, accuracies_support = self._adapt(
            images_support, 
            labels_support, 
            local_parameters
        )

        return local_parameters, accuracies_support


    def _outer_step(self, task_batch):
        """ Performs an outer step.
        
        Args:
            task_batch (dict[str, Tensor]): a batch of tasks
        Returns:
            outer_loss (torch.Tensor): mean outer loss
            accuracies_support (np.array): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """

        outer_loss_batch, accuracies_support_batch, accuracies_query_batch = [], [], []

        for task in task_batch:
            # Get task data
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)

            # Perform inner loop adaptation
            local_parameters, accuracies_support = self._inner_loop(images_support, labels_support)
            accuracies_support_batch.append(accuracies_support)

            # Compute outer loss and accuracy on query set
            loss_query, accuracy_query = self._classify_feats(
                images_query, 
                labels_query, 
                local_parameters
            )
            outer_loss_batch.append(loss_query)
            accuracies_query_batch.append(accuracy_query)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(accuracies_support_batch, axis=0)
        accuracy_query = np.mean(accuracies_query_batch)

        return outer_loss, accuracies_support, accuracy_query


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
            outer_loss, accuracies_support, accuracy_query = self._outer_step(task_batch)
            outer_loss.backward()
            self._optimizer.step()

            outer_loss = outer_loss.item()

            # Log training metrics
            if train_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {train_step}: '
                    f'loss: {outer_loss:.3f}, '
                    f'pre-adaptation accuracy support: {accuracies_support[0]:.3f}, '
                    f'post-adaptation accuracy support: {accuracies_support[-1]:.3f}, '
                    f'post-adaptation accuracy query: {accuracy_query:.3f}'
                )

                metric_pairs = {
                    'loss/train': outer_loss,
                    'pre_adapt_accuracy_support/train': accuracies_support[0],
                    'post_adapt_accuracy_support/train': accuracies_support[-1],
                    'post_adapt_accuracy_query/train': accuracy_query
                }
                tensorboard_writer(writer, metric_pairs, train_step)

            # Evaluate on validation tasks
            if train_step % VAL_INTERVAL == 0:
                losses, accuracies_support, accuracies_query = self.test(val_loader)

                loss = np.mean(losses)
                accuracies_support = np.mean(accuracies_support, axis=0)
                accuracy_query = np.mean(accuracies_query)

                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation accuracy support: {accuracies_support[0]:.3f}, '
                    f'post-adaptation accuracy support: {accuracies_support[-1]:.3f}, '
                    f'post-adaptation accuracy query: {accuracy_query:.3f}'
                )

                metric_pairs = {
                    'loss/val': loss,
                    'pre_adapt_accuracy_support/val': accuracies_support[0],
                    'post_adapt_accuracy_support/val': accuracies_support[-1],
                    'post_adapt_accuracy_query/val': accuracy_query
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
            losses (list): losses for each task
            accuracies_support (list): accuracies on support set for each task
            accuracies_query (list): accuracies on query set for each task
        """

        losses, accuracies_support, accuracies_query = [], [], []

        for task_batch in test_loader:
            outer_loss, accuracy_support, accuracy_query = self._outer_step(task_batch)
            
            losses.append(outer_loss.item())
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
    