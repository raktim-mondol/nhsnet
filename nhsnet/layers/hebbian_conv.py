import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianConv2d(nn.Conv2d):
    """Convolutional layer with Hebbian learning updates"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 hebbian_lr=0.01, decay_factor=0.1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.hebbian_lr = hebbian_lr
        self.decay_factor = decay_factor
        self.register_buffer('hebbian_traces', torch.zeros_like(self.weight))

    def forward(self, x):
        output = super().forward(x)
        
        if self.training:
            # Compute local activations for Hebbian update
            with torch.no_grad():
                # Unfold input for correlation computation
                unfolded = F.unfold(x, self.kernel_size)
                # Compute correlation between input and output
                pre_synaptic = unfolded.transpose(1, 2)
                post_synaptic = output.flatten(2)
                
                # Hebbian update
                correlation = torch.bmm(pre_synaptic.transpose(1, 2), post_synaptic)
                hebbian_update = self.hebbian_lr * (
                    correlation.mean(0) - self.decay_factor * self.weight.view(self.out_channels, -1)
                )
                
                # Update traces
                self.hebbian_traces.add_(hebbian_update.view_as(self.weight))
                
                # Apply traces to weights
                self.weight.data.add_(self.hebbian_traces)
                
        return output