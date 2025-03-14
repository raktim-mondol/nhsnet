import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class HebbianConv2d(nn.Conv2d):
    """Convolutional layer with Hebbian learning updates"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 hebbian_lr=0.01, decay_factor=0.1, max_update_norm=0.05, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.hebbian_lr = hebbian_lr
        self.decay_factor = decay_factor
        self.max_update_norm = max_update_norm  # Maximum norm for Hebbian updates
        self.register_buffer('hebbian_traces', torch.zeros_like(self.weight))
        self.kernel_size = _pair(kernel_size)
        
        # Initialize weights properly
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # Get device from input tensor
        device = x.device
        
        # Ensure weights and traces are on the same device as input
        if self.weight.device != device:
            self.weight = self.weight.to(device)
            self.hebbian_traces = self.hebbian_traces.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        
        # Check for input channel mismatch and fix if needed
        if x.size(1) != self.in_channels:
            # Create a new layer with correct input channels
            new_layer = HebbianConv2d(
                x.size(1), self.out_channels, self.kernel_size[0],
                hebbian_lr=self.hebbian_lr, decay_factor=self.decay_factor,
                max_update_norm=self.max_update_norm,
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups,
                bias=self.bias is not None
            ).to(device)
            # Transfer existing weights where possible
            with torch.no_grad():
                min_channels = min(x.size(1), self.in_channels)
                new_layer.weight.data[:, :min_channels].copy_(
                    self.weight.data[:, :min_channels]
                )
                if self.bias is not None:
                    new_layer.bias.data.copy_(self.bias.data)
                new_layer.hebbian_traces.data[:, :min_channels].copy_(
                    self.hebbian_traces.data[:, :min_channels]
                )
            # Replace self with new layer
            self.in_channels = new_layer.in_channels
            self.weight = new_layer.weight
            self.hebbian_traces = new_layer.hebbian_traces
            if self.bias is not None:
                self.bias = new_layer.bias
        
        # Forward pass
        output = super().forward(x)
        
        if self.training:
            # Compute local activations for Hebbian update
            with torch.no_grad():
                # Check for NaN values
                if torch.isnan(x).any() or torch.isnan(output).any():
                    return output
                
                # Apply Hebbian updates with probability 0.5 to reduce instability
                if torch.rand(1).item() < 0.5:
                    return output
                
                # Unfold input for correlation computation
                batch_size = x.size(0)
                unfolded = F.unfold(x, self.kernel_size, 
                                  stride=self.stride,
                                  padding=self.padding,
                                  dilation=self.dilation)
                
                # Reshape unfolded input for correlation
                n_locations = unfolded.size(-1)
                pre_synaptic = unfolded.view(batch_size, self.in_channels * self.kernel_size[0] * self.kernel_size[1], n_locations)
                post_synaptic = output.view(batch_size, self.out_channels, -1)
                
                # Compute correlation between pre and post synaptic activations
                correlation = torch.zeros_like(self.weight).view(self.out_channels, -1)
                for i in range(batch_size):
                    correlation.add_(torch.mm(post_synaptic[i], pre_synaptic[i].t()))
                correlation.div_(batch_size * n_locations)
                
                # Hebbian update with gradient clipping
                hebbian_update = self.hebbian_lr * (
                    correlation - self.decay_factor * self.weight.view(self.out_channels, -1)
                )
                
                # Clip the update to prevent exploding values
                update_norm = torch.norm(hebbian_update)
                if update_norm > self.max_update_norm:
                    hebbian_update = hebbian_update * (self.max_update_norm / update_norm)
                
                # Update traces with exponential moving average
                self.hebbian_traces.view(self.out_channels, -1).mul_(0.9).add_(hebbian_update, alpha=0.1)
                
                # Apply traces to weights with normalization
                self.weight.data.add_(self.hebbian_traces)
                
                # Normalize weights to prevent explosion
                with torch.no_grad():
                    weight_norm = torch.norm(self.weight.data.view(self.out_channels, -1), dim=1, keepdim=True)
                    weight_norm = weight_norm.view(self.out_channels, 1, 1, 1)
                    self.weight.data = self.weight.data / (weight_norm + 1e-6)
                
        return output