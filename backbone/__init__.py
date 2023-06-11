######################################################################
# Model backbone used for both Meta-learning and Transfer-learning.  #
#                                                                    #
# The network consists of four convolutional blocks, each comprising #
# a convolution layer, a batch normalization layer, ReLU activation, # 
# and 2x2 max pooling for downsampling. There is an additional       #
# flattening operation at the end.                                   #
#                                                                    #
# Note that unlike conventional use, batch normalization is always   #
# done with batch statistics, regardless of whether we are training  # 
# or evaluating. This technically makes meta-learning transductive,  # 
# as opposed to inductive.                                           #
######################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
STRIDE = 1
PADDING = "same"


class Backbone:
    @staticmethod
    def get_network(num_outputs, device):
        """
        Returns initialized parameters for the network.

        Args:
            device (str): device to use for computation
            num_outputs (int): dimensionality of output space
        Returns:
            parameters (dict[str, Tensor]): parameters to use
        """
        
        parameters = {}

        # Construct feature extractor
        in_channels = NUM_INPUT_CHANNELS
        for i in range(NUM_CONV_LAYERS):
            parameters[f'conv{i}'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    in_channels,
                    KERNEL_SIZE,
                    KERNEL_SIZE,
                    requires_grad=True,
                    device=device
                )
            )
            parameters[f'b{i}'] = nn.init.zeros_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    requires_grad=True,
                    device=device
                )
            )
            in_channels = NUM_HIDDEN_CHANNELS

        # Construct linear head layer
        parameters[f'w{NUM_CONV_LAYERS}'] = nn.init.xavier_uniform_(
            torch.empty(
                num_outputs,
                NUM_HIDDEN_CHANNELS,
                requires_grad=True,
                device=device
            )
        )
        parameters[f'b{NUM_CONV_LAYERS}'] = nn.init.zeros_(
            torch.empty(
                num_outputs,
                requires_grad=True,
                device=device
            )
        )

        return parameters


    @staticmethod
    def forward(images, parameters=None):
        """Computes predicted classification logits.

        Args:
            images (Tensor): batch of images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for
                the computation
        Returns:
            a resulting Tensor consisting of a batch of logits
                shape (num_images, num_outputs)
        """
        
        x = images

        for i in range(NUM_CONV_LAYERS):
            x = F.conv2d(
                input=x,
                weight=parameters[f'conv{i}'],
                bias=parameters[f'b{i}'],
                stride=STRIDE,
                padding=PADDING
            )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

        x = torch.mean(x, dim=[2, 3])

        return F.linear(
            input=x,
            weight=parameters[f'w{NUM_CONV_LAYERS}'],
            bias=parameters[f'b{NUM_CONV_LAYERS}']
        )
    