import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class HebbianConv2d(nn.Conv2d):
    """Convolutional layer with Hebbian learning updates"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 hebbian_lr=0.01, decay_factor=0.1, max_update_norm=0.05,
                 stabilization_factor=0.01, trace_decay=0.9, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.hebbian_lr = hebbian_lr
        self.decay_factor = decay_factor
        self.max_update_norm = max_update_norm  # Maximum norm for Hebbian updates
        self.stabilization_factor = stabilization_factor  # L2 regularization factor
        self.trace_decay = trace_decay  # Decay rate for hebbian traces
        
        # Register buffers for Hebbian learning
        self.register_buffer('hebbian_traces', torch.zeros_like(self.weight))
        self.register_buffer('weight_momentum', torch.zeros_like(self.weight))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long))
        
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
            self.weight_momentum = self.weight_momentum.to(device)
            self.update_count = self.update_count.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        
        # Check for input channel mismatch and fix if needed
        if x.size(1) != self.in_channels:
            # Create a new layer with correct input channels
            new_layer = HebbianConv2d(
                x.size(1), self.out_channels, self.kernel_size[0],
                hebbian_lr=self.hebbian_lr, decay_factor=self.decay_factor,
                max_update_norm=self.max_update_norm,
                stabilization_factor=self.stabilization_factor,
                trace_decay=self.trace_decay,
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
                new_layer.weight_momentum.data[:, :min_channels].copy_(
                    self.weight_momentum.data[:, :min_channels]
                )
                new_layer.update_count.data.copy_(self.update_count.data)
            # Replace self with new layer
            self.in_channels = new_layer.in_channels
            self.weight = new_layer.weight
            self.hebbian_traces = new_layer.hebbian_traces
            self.weight_momentum = new_layer.weight_momentum
            self.update_count = new_layer.update_count
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
                
                # Apply Hebbian updates with adaptive probability based on update count
                # More frequent updates early in training, less frequent later
                update_prob = max(0.1, min(0.5, 1.0 / (1.0 + 0.01 * self.update_count.item())))
                if torch.rand(1).item() > update_prob:
                    return output
                
                # Increment update counter
                self.update_count.add_(1)
                
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
                
                # Apply activation function to post-synaptic activations for non-linear Hebbian learning
                post_synaptic = torch.sigmoid(post_synaptic)
                
                # Compute correlation between pre and post synaptic activations
                correlation = torch.zeros_like(self.weight).view(self.out_channels, -1)
                for i in range(batch_size):
                    # Compute correlation with normalization
                    pre_norm = F.normalize(pre_synaptic[i], dim=0)
                    post_norm = F.normalize(post_synaptic[i], dim=1)
                    correlation.add_(torch.mm(post_norm, pre_norm.t()))
                correlation.div_(batch_size)
                
                # Reshape correlation to match weight dimensions
                correlation = correlation.view_as(self.weight)
                
                # Compute L2 regularization term
                l2_reg = self.stabilization_factor * self.weight
                
                # Hebbian update with weight decay and L2 regularization
                hebbian_update = self.hebbian_lr * (
                    correlation - self.decay_factor * self.weight - l2_reg
                )
                
                # Clip the update to prevent exploding values
                update_norm = torch.norm(hebbian_update)
                if update_norm > self.max_update_norm:
                    hebbian_update = hebbian_update * (self.max_update_norm / update_norm)
                
                # Update momentum with exponential moving average
                self.weight_momentum.mul_(0.9).add_(hebbian_update, alpha=0.1)
                
                # Update traces with exponential moving average
                self.hebbian_traces.mul_(self.trace_decay).add_(hebbian_update, alpha=1-self.trace_decay)
                
                # Apply traces to weights with adaptive learning rate
                adaptive_lr = 1.0 / (1.0 + 0.001 * self.update_count.item())
                self.weight.data.add_(self.weight_momentum, alpha=adaptive_lr)
                
                # Normalize weights periodically to prevent explosion (every 10 updates)
                if self.update_count.item() % 10 == 0:
                    weight_norm = torch.norm(self.weight.data.view(self.out_channels, -1), dim=1, keepdim=True)
                    weight_norm = weight_norm.view(self.out_channels, 1, 1, 1)
                    self.weight.data = self.weight.data / (weight_norm + 1e-6)
                
                # Zero out any NaN values that might have occurred
                self.weight.data = torch.nan_to_num(self.weight.data, 0.0)
                
        return output