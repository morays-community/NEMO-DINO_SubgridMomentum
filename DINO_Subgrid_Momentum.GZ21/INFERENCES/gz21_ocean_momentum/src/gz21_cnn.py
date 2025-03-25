# 2023-05-12 raehik: try to replace with fully_conv_net. should be identical.

# -*- coding: utf-8 -*-
"""
Modified version of 
"""
import torch
from torch import nn

# THIS IS THE MODEL USED IN THE FINAL PAPER
class FullyCNN(nn.Sequential):
    """
    Fully Convolutional Neural Net used for modelling ocean momentum.

    Attributes
    ----------
    n_in_channels : int
        number of input chanels to model, 2 for the two surface velocity
        components
    n_out_channels : int
        number of output channels from model, 4 with 2 per component
        of the subgrid forcing
    padding : str
        padding argument passed on to Conv2d layers
    batch_norm : bool
        whether to normalise batches

    Methods
    -------
    final_transformation
    forward
    """

    def __init__(
        self,
        n_in_channels: int = 2,
        n_out_channels: int = 4,
        padding=None,
        batch_norm=False,
    ):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_init_mode = 'zeros'
            padding_init = padding_5
        elif padding == "same":
            padding_5 = 2
            padding_3 = 1
            padding_init = padding_5
            padding_init_mode = 'zeros'
        elif padding == "init_zeros" : 
            padding_5 = 0
            padding_3 = 0
            padding_init = 10
            padding_init_mode = 'zeros'
        else:
            raise ValueError("Unknow value for padding parameter.")

        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 128, 5, padding=padding_init, padding_mode=padding_init_mode)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(128, 64, 5, padding=padding_5)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(64, 32, 3, padding=padding_3)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(32, n_out_channels, 3, padding=padding_3)

        nn.Sequential.__init__(
            self, *block1, *block2, *block3, *block4, *block5, *block6, *block7, conv8
        )


    def forward(self, x):
        """
        Forward method for the model.

        Parameters
        ----------
        x : tensor
            Input coarse velocities, with shape (N, C, H, W)

        Returns
        -------
        self.final_transformation(x) : tensor
            input with final_transformation operation performed
        """
        x = super().forward(x)
        return x # Already applying the final_transformation as it's a building block of the sequential model

    def _make_subblock(self, conv):
        """
        Building unit for a convolution subblock followed
        by a ReLU.

        Parameters
        ----------
        conv : Conv2d
            Conv2d Module instance

        Returns
        -------
        subblock : Module
            A Module composed of a Conv2d followed by a ReLU,
            potentially with Batchnorm if activated via
            the batchnorm attribute.
        """
        subblock = [conv, nn.ReLU()]
        if self.batch_norm:
            subblock.append(nn.BatchNorm2d(conv.out_channels))
        return subblock