import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from .hebbian_conv import HebbianConv2d

class DynamicNeurogenesisModule(nn.Module):
    """Implements dynamic neuron addition based on activation statistics"""
    def __init__(self, 
                 initial_neurons, 
                 max_neurons,
                 activation_threshold=0.1,
                 growth_factor=0.1,
                 pca_components=10,
                 weight_scale=0.01,
                 history_size=5):
        super().__init__()
        self.initial_neurons = initial_neurons
        self.max_neurons = max_neurons
        self.activation_threshold = activation_threshold
        self.growth_factor = growth_factor
        self.pca_components = pca_components
        self.weight_scale = weight_scale
        self.history_size = history_size
        
        self.activation_history = []
        self.neuron_count = initial_neurons
        self.last_expansion_step = 0
        self.min_steps_between_expansions = 100  # Minimum steps between expansions
        self.current_step = 0
        
    def compute_activation_statistics(self, activations):
        """Compute mean activation and identify underactivated regions"""
        # Handle NaN values
        if torch.isnan(activations).any():
            return torch.zeros_like(activations.mean(dim=[0, 2, 3])), torch.zeros_like(activations.mean(dim=[0, 2, 3])).bool()
            
        # Ensure we're working with the right shape
        if len(activations.shape) == 4:  # [B, C, H, W]
            # Add epsilon to prevent division by zero
            mean_activation = torch.mean(activations, dim=[0, 2, 3])  # [C]
            std_activation = torch.std(activations, dim=[0, 2, 3]) + 1e-6
            
            # Normalize activations
            mean_activation = mean_activation / std_activation
            
            # Clear history if tensor size changes
            if len(self.activation_history) > 0 and self.activation_history[0].size(0) != mean_activation.size(0):
                print(f"Clearing activation history due to size change: {self.activation_history[0].size(0)} -> {mean_activation.size(0)}")
                self.activation_history = []
            
            # Store activation history for more stable decisions
            if len(self.activation_history) >= self.history_size:
                self.activation_history.pop(0)
            self.activation_history.append(mean_activation.detach().clone())
            
            # Average over history for more stable decisions
            if len(self.activation_history) > 0:
                mean_activation = torch.stack(self.activation_history).mean(dim=0)
        else:
            mean_activation = torch.mean(activations, dim=0)  # [C]
            std_activation = torch.std(activations, dim=0) + 1e-6
            mean_activation = mean_activation / std_activation
            
            # Clear history if tensor size changes
            if len(self.activation_history) > 0 and self.activation_history[0].size(0) != mean_activation.size(0):
                print(f"Clearing activation history due to size change: {self.activation_history[0].size(0)} -> {mean_activation.size(0)}")
                self.activation_history = []
            
            # Store activation history
            if len(self.activation_history) >= self.history_size:
                self.activation_history.pop(0)
            self.activation_history.append(mean_activation.detach().clone())
            
            # Average over history
            if len(self.activation_history) > 0:
                mean_activation = torch.stack(self.activation_history).mean(dim=0)
            
        # Increment step counter
        self.current_step += 1
        
        # Check if we should consider expansion based on time since last expansion
        if self.current_step - self.last_expansion_step < self.min_steps_between_expansions:
            return mean_activation, torch.zeros_like(mean_activation).bool()
            
        # Identify under-activated neurons
        under_activated = mean_activation < self.activation_threshold
        return mean_activation, under_activated
        
    def _normalize_weights(self, weights):
        """Apply weight normalization"""
        norm = torch.norm(weights.reshape(weights.size(0), -1), dim=1, keepdim=True)
        norm = norm.reshape(-1, 1, 1, 1) if len(weights.shape) == 4 else norm.reshape(-1, 1)
        return weights / (norm + 1e-6) * self.weight_scale
        
    def generate_new_neurons(self, layer, activation_patterns):
        """Generate new neurons using PCA on activation patterns"""
        if self.neuron_count >= self.max_neurons:
            return None
            
        # Handle NaN values
        if torch.isnan(activation_patterns).any():
            return None
            
        # Flatten activation patterns if needed
        if len(activation_patterns.shape) > 2:
            b, c, h, w = activation_patterns.shape
            # Use spatial information by sampling patches
            patches = F.unfold(activation_patterns, kernel_size=3, padding=1, stride=2)
            # Reshape to [B, C*K*K, L] where L is the number of patches
            b, n_features, n_patches = patches.shape
            # Reshape to [B*L, C*K*K] for PCA
            patches = patches.permute(0, 2, 1).reshape(-1, n_features)
            activation_patterns = patches
            
        # Remove NaN values and normalize
        activation_patterns = torch.nan_to_num(activation_patterns, 0.0)
        activation_patterns = F.normalize(activation_patterns, dim=1)
            
        # Perform PCA on activation patterns
        try:
            U, S, V = torch.pca_lowrank(activation_patterns, q=min(self.pca_components, activation_patterns.size(1)))
        except RuntimeError:
            return None
        
        # Number of neurons to add (ensure we don't exceed max_neurons)
        n_new = int(self.growth_factor * self.neuron_count)
        n_new = min(n_new, self.max_neurons - self.neuron_count)
        
        # If no new neurons can be added, return None
        if n_new <= 0:
            return None
            
        # Update last expansion step
        self.last_expansion_step = self.current_step
            
        if isinstance(layer, nn.Conv2d):
            new_weights = self._generate_conv_weights(layer, V[:n_new])
        else:
            new_weights = self._generate_linear_weights(layer, V[:n_new])
            
        # Apply weight normalization
        new_weights = self._normalize_weights(new_weights)
            
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
        
        # Apply orthogonalization to ensure diversity
        if new_weights.size(0) > 1:
            new_weights = self._orthogonalize_weights(new_weights)
        
        return new_weights
    
    def _orthogonalize_weights(self, weights):
        """Apply Gram-Schmidt orthogonalization to ensure diverse weights"""
        # Reshape to [N, -1] for orthogonalization
        n, c, h, w = weights.shape
        weights_flat = weights.reshape(n, -1)
        
        # Apply Gram-Schmidt process
        for i in range(1, n):
            for j in range(i):
                # Project weights[i] onto weights[j]
                projection = torch.sum(weights_flat[i] * weights_flat[j]) / (torch.sum(weights_flat[j] * weights_flat[j]) + 1e-8)
                # Subtract projection
                weights_flat[i] = weights_flat[i] - projection * weights_flat[j]
            
            # Normalize
            norm = torch.norm(weights_flat[i]) + 1e-8
            weights_flat[i] = weights_flat[i] / norm
        
        # Reshape back to original shape
        return weights_flat.reshape(n, c, h, w)
        
    def _generate_linear_weights(self, layer, basis_vectors):
        """Generate new linear layer weights"""
        # Project basis vectors if needed
        if basis_vectors.size(1) != layer.in_features:
            projection_matrix = torch.randn(basis_vectors.size(1), layer.in_features, device=basis_vectors.device)
            basis_vectors = torch.mm(basis_vectors, projection_matrix)
        
        # Normalize the weights
        new_weights = F.normalize(basis_vectors, dim=1)
        
        # Apply orthogonalization for diversity
        if new_weights.size(0) > 1:
            # Reshape to [N, -1] for orthogonalization
            n, f = new_weights.shape
            
            # Apply Gram-Schmidt process
            for i in range(1, n):
                for j in range(i):
                    # Project weights[i] onto weights[j]
                    projection = torch.sum(new_weights[i] * new_weights[j]) / (torch.sum(new_weights[j] * new_weights[j]) + 1e-8)
                    # Subtract projection
                    new_weights[i] = new_weights[i] - projection * new_weights[j]
                
                # Normalize
                norm = torch.norm(new_weights[i]) + 1e-8
                new_weights[i] = new_weights[i] / norm
        
        return new_weights
        
    def expand_layer(self, layer, activation_patterns):
        """Expand layer with new neurons"""
        # Ensure we're on the same device as the input
        device = activation_patterns.device
        
        # Check for channel mismatch between activation patterns and layer
        if isinstance(layer, nn.Conv2d) and activation_patterns.size(1) != layer.in_channels:
            print(f"Skipping expansion due to channel mismatch: activation={activation_patterns.size(1)}, layer={layer.in_channels}")
            return layer
            
        new_weights = self.generate_new_neurons(layer, activation_patterns)
        if new_weights is None:
            return layer
            
        # Get the device of the current layer
        layer_device = layer.weight.device
        
        # Ensure layer is on the same device as input
        if layer_device != device:
            layer = layer.to(device)
            
        if isinstance(layer, nn.Conv2d):
            # Calculate new output channels
            new_out_channels = layer.out_channels + new_weights.size(0)
            
            # Check if expansion would exceed max neurons
            if new_out_channels > self.max_neurons:
                return layer
                
            # Create new layer with correct dimensions
            expanded_layer = nn.Conv2d(
                layer.in_channels,  # Keep original input channels
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
                
            # Add new weights with proper normalization
            new_weights = self._normalize_weights(new_weights)
            expanded_layer.weight.data[layer.out_channels:] = new_weights
            
            # If it's a HebbianConv2d, convert the expanded layer
            if isinstance(layer, HebbianConv2d):
                hebbian_layer = HebbianConv2d(
                    layer.in_channels,
                    new_out_channels,
                    layer.kernel_size[0],
                    stride=layer.stride,
                    padding=layer.padding,
                    hebbian_lr=layer.hebbian_lr,
                    decay_factor=layer.decay_factor,
                    max_update_norm=layer.max_update_norm if hasattr(layer, 'max_update_norm') else 0.05
                ).to(device)
                hebbian_layer.weight.data = expanded_layer.weight.data.clone()
                if layer.bias is not None:
                    hebbian_layer.bias.data = expanded_layer.bias.data.clone()
                # Copy hebbian traces
                if hasattr(layer, 'hebbian_traces'):
                    hebbian_layer.hebbian_traces = torch.zeros_like(hebbian_layer.weight, device=device)
                    hebbian_layer.hebbian_traces[:layer.out_channels] = layer.hebbian_traces
                expanded_layer = hebbian_layer
            
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
                
            # Add new weights with proper normalization
            new_weights = self._normalize_weights(new_weights)
            expanded_layer.weight.data[layer.out_features:] = new_weights
            
        return expanded_layer