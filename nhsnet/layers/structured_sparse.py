import torch
import torch.nn as nn
import numpy as np

class StructuredSparseConv2d(nn.Conv2d):
    """Convolutional layer with structured sparsity pattern"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 sparsity_ratio=0.5, pattern='hexagonal', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.sparsity_ratio = sparsity_ratio
        self.pattern = pattern
        self.register_buffer('mask', self._create_sparse_mask())
        
        # Apply mask to weights
        self.weight.data.mul_(self.mask)
        
    def _create_sparse_mask(self):
        if self.pattern == 'hexagonal':
            return self._create_hexagonal_mask()
        else:
            return self._create_random_mask()
            
    def _create_hexagonal_mask(self):
        mask = torch.ones_like(self.weight)
        k = self.kernel_size[0]
        center = k // 2
        
        for i in range(k):
            for j in range(k):
                if (i + j) % 2 == 0:  # Hexagonal pattern
                    mask[:, :, i, j] = 0
                    
        return mask
        
    def forward(self, x):
        # Ensure weights remain sparse
        self.weight.data.mul_(self.mask)
        return super().forward(x)