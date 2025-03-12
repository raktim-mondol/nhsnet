import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class HebbianConv2d(nn.Conv2d):
    """Convolutional layer with Hebbian learning updates"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 hebbian_lr=0.01, decay_factor=0.1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.hebbian_lr = hebbian_lr
        self.decay_factor = decay_factor
        self.register_buffer('hebbian_traces', torch.zeros_like(self.weight))
        self.kernel_size = _pair(kernel_size)

    def forward(self, x):
        output = super().forward(x)
        
        if self.training:
            # Compute local activations for Hebbian update
            with torch.no_grad():
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
                
                # Hebbian update
                hebbian_update = self.hebbian_lr * (
                    correlation - self.decay_factor * self.weight.view(self.out_channels, -1)
                )
                
                # Update traces
                self.hebbian_traces.view(self.out_channels, -1).add_(hebbian_update)
                
                # Apply traces to weights
                self.weight.data.add_(self.hebbian_traces)
                
        return output