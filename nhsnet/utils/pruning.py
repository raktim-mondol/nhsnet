import torch
import torch.nn as nn
import numpy as np

class AdaptiveSynapticPruning:
    """Implements adaptive synaptic pruning based on weight magnitude"""
    def __init__(self, 
                 model, 
                 pruning_ratio=0.1,
                 pruning_interval=1000,
                 min_weights=0.1):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.pruning_interval = pruning_interval
        self.min_weights = min_weights
        self.step_counter = 0
        self.masks = self._initialize_masks()
        
    def _initialize_masks(self):
        """Initialize binary masks for all weights"""
        masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                masks[name] = torch.ones_like(param)
        return masks
        
    def _compute_threshold(self, weights):
        """Compute pruning threshold based on weight distribution"""
        abs_weights = torch.abs(weights)
        return torch.quantile(abs_weights, self.pruning_ratio)
        
    def step(self):
        """Perform one step of pruning if interval is reached"""
        self.step_counter += 1
        
        if self.step_counter % self.pruning_interval == 0:
            self._prune_weights()
            
    def _prune_weights(self):
        """Prune weights below threshold"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Compute pruning threshold
                threshold = self._compute_threshold(param.data)
                
                # Update mask
                new_mask = (torch.abs(param.data) > threshold).float()
                
                # Ensure minimum number of weights remain
                if torch.sum(new_mask) < self.min_weights * new_mask.numel():
                    continue
                    
                self.masks[name] = new_mask
                
                # Apply mask
                param.data.mul_(new_mask)
                
    def apply_masks(self):
        """Apply existing masks to weights"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and name in self.masks:
                param.data.mul_(self.masks[name])