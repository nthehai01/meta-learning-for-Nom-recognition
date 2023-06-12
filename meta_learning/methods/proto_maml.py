#####################################################
# Implementation of Proto-MAML for Nôm recognition. #
#####################################################


import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torch.nn.functional as F

from meta_learning.methods.maml import MAML
from meta_learning.methods.protonet import ProtoNet


OUTPUT_WEIGHTS_NAME = 'w_out'
OUTPUT_BIAS_NAME = 'b_out'


class ProtoMAML(MAML):
    """Trains and assesses a Proto-MAML."""

    def __init__(self, 
                 feat_dim, 
                 num_inner_steps, 
                 inner_lr, 
                 outer_lr, 
                 log_dir):
        """ Initializes Proto-MAML.

        Args: 
            feat_dim (int): dimensionality of feature space,
                unlike MAML, this does not have to be the num_way 
            num_inner_steps (int): number of inner steps
            inner_lr (float): inner learning rate
            outer_lr (float): outer learning rate
            log_dir (str): directory to save or load model checkpoints
        """

        super().__init__(
            feat_dim, 
            num_inner_steps, 
            inner_lr, 
            outer_lr, 
            log_dir
        )


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

        x = super()._forward(images, parameters)

        # Pass through additional output fc layer
        if (OUTPUT_WEIGHTS_NAME in parameters.keys()) and (OUTPUT_BIAS_NAME in parameters.keys()):
            x = F.linear(x, parameters[OUTPUT_WEIGHTS_NAME], parameters[OUTPUT_BIAS_NAME])

        return x
    

    def _compute_prototypes(self, images, labels):
        """ Computes the prototypes for each class.

        Args:
            images (torch.Tensor): images to compute prototypes for
                shape (n, c, h, w)
            labels (torch.Tensor): labels for each image
                shape (n,)
        Returns:
            prototypes (torch.Tensor): class prototypes as a Tensor
                shape (num_way, feat_dim)
        """

        with torch.no_grad():
            logits = self._forward(images, self._network)  # shape (n, feat_dim)
            prototypes = ProtoNet.compute_prototypes(logits, labels)

        return prototypes
    

    def _init_output_layer(self, prototypes):
        """ Initializes output layer with prototype-based initialization.

        Args:
            prototypes (torch.Tensor): prototypes to use for initialization
                shape (num_way, feat_dim)
        Returns:
            output_layer (dict[str, Tensor]): output layer parameters 
                with weights and biases
        """
        
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2

        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        output_layer = {
            OUTPUT_WEIGHTS_NAME: output_weight, 
            OUTPUT_BIAS_NAME: output_bias
        }

        return output_layer
    

    def _init_local_parameters(self, images_support, labels_support):
        """ Initializes task-specific parameters by cloning the model's meta-params,
            as well as adding new output layer parameters.

        Args:
            images_support (torch.Tensor): support images
                shape (num_way * num_shot, c, h, w)
            labels_support (torch.Tensor): support labels
                shape (num_way * num_shot,)
        Returns:
            local_parameters (dict[str, Tensor]): local parameters
        """

        # Determine prototype initialization
        prototypes = self._compute_prototypes(images_support, labels_support)

        # Create output layer weights with prototype-based initialization
        output_layer = self._init_output_layer(prototypes)

        # Make a clone of the meta parameters
        local_parameters = {
            k: torch.clone(v)
            for k, v in self._network.items()
        }

        # Append a clone of the output layer parameters
        local_parameters.update({
            k: torch.clone(v)
            for k, v in output_layer.items()
        })

        return local_parameters
    

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
        local_parameters = self._init_local_parameters(images_support, labels_support)

        # Perform adaptation
        local_parameters, accuracies_support = self._adapt(
            images_support, 
            labels_support, 
            local_parameters
        )

        return local_parameters, accuracies_support
          