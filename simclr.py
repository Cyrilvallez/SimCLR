#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:28:09 2022

@author: cyrilvallez
"""

import torch
import torch.nn as nn


class SimCLR(nn.Module):
    """
    Module representing SimCLR. 

    Parameters
    ----------
    encoder : torch.nn.Module
        Model representing the encoder.
    contrastive_head : torch.nn.Module
        Model representing the contrastive/projective head.
        
    """
    
    def __init__(self, encoder, contrastive_head):
        
        super().__init__()
        self.encoder = encoder
        self.contrastive_head = contrastive_head
        
    def forward(self, x1, x2):
        """
        Implements the forward.

        Parameters
        ----------
        x1 : Tensor
            The batch of first data augmentation.
        x2 : Tensor
            The batch of second data augmentation.

        Returns
        -------
        z1 : Tensor
            Output corresponding to 1st data augmentation.
        z2 : Tensor
            Output corresponding to 2nd data augmentation.

        """
    
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
    
        z1 = self.contrastive_head(h1)
        z2 = self.contrastive_head(h2)
        
        return z1, z2
        
        
    def save(self, path):
        """
        Save the model, separating the encoder and the head.

        Parameters
        ----------
        path : str
            Path where to save the model.

        Returns
        -------
        None.

        """
        
        state = {'encoder': self.encoder.state_dict(),
                 'head': self.contrastive_head.state_dict()}
        torch.save(state, path)
        
        
    @staticmethod
    def load(path, encoder_arch, head_arch, map_location=None):
        """
        Easily load a SimCLR module saved by the save() method.

        Parameters
        ----------
        path : str
            Path to the saved SimCLR model. 
        encoder_arch : torch.nn.Module
            Instance of the model representing the encoder. This should be the same 
            as what you previously used for correct mapping of the weights.
        head_arch : torch.nn.Module
            Instance of the model representing the head. This should be the same 
            as what you previously used for correct mapping of the weights.
        map_location : function, torch device, str or dict
            Specify how to remap storage locations. See torch.load for more details.
            The default is None.

        Returns
        -------
        torch.nn.Module
            The simCLR module.

        """
        
        checkpoint = torch.load(path, map_location=map_location)
        encoder_arch.load_state_dict(checkpoint['encoder'])
        head_arch.load_state_dict(checkpoint['head'])
        
        return SimCLR(encoder_arch, head_arch)
    
        
    