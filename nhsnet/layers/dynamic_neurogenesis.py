import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init

class DynamicNeurogenesisModule(nn.Module):
    """Implements dynamic neuron addition based on activation statistics"""
    def __init__(self, 
                 initial_neurons, 
                 max_neurons,
                 activation_threshold=0.1,
                 growth_factor=0.1,
                 pca_components=10):
        super().__init__()
        self.initial_neurons = initial_neurons
        self.max_neurons = max_neurons
        self.activation_threshold = activation_threshold
        self.growth_factor = growth_factor
        self.pca_components = pca_components
        
        self.activation_history = []
        self.neuron_count = initial_neurons
        
    def compute_activation_statistics(self, activations):
        """Compute mean activation and identify underactivated regions"""
        # Ensure we're working with the right shape
        if len(activations.shape) == 4:  # [B, C, H, W]
            mean_activation = torch.mean(activations, dim=[0, 2, 3])  # [C]
        else:
            mean_activation = torch.mean(activations, dim=0)  # [C]
            
        under_activated = mean_activation < self.activation_threshold
        return mean_activation, under_activated
        
    def generate_new_neurons(self, layer, activation_patterns):
        """Generate new neurons using PCA on activation patterns"""
        if self.neuron_count >= self.max_neurons:
            return None
            
        # Flatten activation patterns if needed
        if len(activation_patterns.shape) > 2:
            b, c, h, w = activation_patterns.shape
            activation_patterns = activation_patterns.view(b, c, -1).mean(dim=2)
            
        # Perform PCA on activation patterns
        U, S, V = torch.pca_lowrank(activation_patterns, q=min(self.pca_components, activation_patterns.size(1)))
        
        # Number of neurons to add (ensure we don't exceed max_neurons)
        n_new = int(self.growth_factor * self.neuron_count)
        n_new = min(n_new, self.max_neurons - self.neuron_count)
        
        # If no new neurons can be added, return None
        if n_new <= 0:
            return None
            
        if isinstance(layer, nn.Conv2d):
            new_weights = self._generate_conv_weights(layer, V[:n_new])
        else:
            new_weights = self._generate_linear_weights(layer, V[:n_new])
            
        self.neuron_count += n_new
        return new_weights
        
    def _generate_conv_weights(self, layer, basis_vectors):
        """Generate new convolutional weights"""
        # Calculate the total number of weights needed
        total_weights = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        
        # Ensure basis vectors have the right size
        if basis_vectors.size(1) != total_weights:
            # Project basis vectors to the right dimension if needed
            projection_matrix = torch.randn(basis_vectors.size(1), total_weights, device=basis_vectors.device)
            basis_vectors = torch.mm(basis_vectors, projection_matrix)
        
        # Normalize the weights
        basis_vectors = F.normalize(basis_vectors, dim=1)
        
        # Reshape to convolutional weights
        new_weights = basis_vectors.view(
            basis_vectors.size(0),  # number of new neurons
            layer.in_channels,
            layer.kernel_size[0],
            layer.kernel_size[1]
        )
        
        return new_weights
        
    def _generate_linear_weights(self, layer, basis_vectors):
        """Generate new linear layer weights"""
        # Project basis vectors if needed
        if basis_vectors.size(1) != layer.in_features:
            projection_matrix = torch.randn(basis_vectors.size(1), layer.in_features, device=basis_vectors.device)
            basis_vectors = torch.mm(basis_vectors, projection_matrix)
        
        # Normalize the weights
        new_weights = F.normalize(basis_vectors, dim=1)
        return new_weights
        
    def expand_layer(self, layer, activation_patterns):
        """Expand layer with new neurons"""
        new_weights = self.generate_new_neurons(layer, activation_patterns)
        if new_weights is None:
            return layer
            
        # Get the device of the current layer
        device = layer.weight.device
            
        if isinstance(layer, nn.Conv2d):
            # Calculate new output channels
            new_out_channels = layer.out_channels + new_weights.size(0)
            
            # Check if expansion would exceed max neurons
            if new_out_channels > self.max_neurons:
                return layer
                
            expanded_layer = nn.Conv2d(
                layer.in_channels,
                new_out_channels,
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=layer.bias is not None
            ).to(device)
            
            # Copy existing weights and biases
            expanded_layer.weight.data[:layer.out_channels] = layer.weight.data
            if layer.bias is not None:
                expanded_layer.bias.data = torch.zeros(new_out_channels, device=device)
                expanded_layer.bias.data[:layer.out_channels] = layer.bias.data
                
            # Add new weights
            expanded_layer.weight.data[layer.out_channels:] = new_weights
            
        else:  # Linear layer
            # Calculate new output features
            new_out_features = layer.out_features + new_weights.size(0)
            
            # Check if expansion would exceed max neurons
            if new_out_features > self.max_neurons:
                return layer
                
            expanded_layer = nn.Linear(
                layer.in_features,
                new_out_features,
                bias=layer.bias is not None
            ).to(device)
            
            expanded_layer.weight.data[:layer.out_features] = layer.weight.data
            if layer.bias is not None:
                expanded_layer.bias.data = torch.zeros(new_out_features, device=device)
                expanded_layer.bias.data[:layer.out_features] = layer.bias.data
                
            expanded_layer.weight.data[layer.out_features:] = new_weights
            
        return expanded_layer