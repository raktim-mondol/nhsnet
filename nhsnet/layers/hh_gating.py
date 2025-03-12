import torch
import torch.nn as nn
import torch.nn.functional as F

class HodgkinHuxleyGating(nn.Module):
    """Implements a simplified Hodgkin-Huxley gating mechanism using an MLP approximation"""
    def __init__(self, channels, beta=0.1, v_threshold=-55.0, v_reset=-70.0):
        super().__init__()
        self.channels = channels
        self.beta = beta
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        # Gating network parameters - operates on channel-wise statistics
        self.gate_network = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels * 3),  # 3 outputs for m, h, and V
        )
        
        # Initialize gating variables
        self.register_buffer('m', torch.ones(channels))
        self.register_buffer('h', torch.ones(channels))
        self.register_buffer('V', torch.ones(channels) * v_reset)
        
    def reset_state(self):
        """Reset gating variables to initial state"""
        self.m.fill_(1.0)
        self.h.fill_(1.0)
        self.V.fill_(self.v_reset)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Compute channel-wise statistics (mean across spatial dimensions)
        x_stats = x.view(batch_size, self.channels, -1).mean(dim=2)  # [B, C]
        
        # Compute gating variables
        gate_output = self.gate_network(x_stats)  # [B, C*3]
        m_new, h_new, v_new = torch.chunk(gate_output, 3, dim=1)  # Each [B, C]
        
        # Update gating variables with temporal dynamics
        self.m = F.sigmoid(m_new.mean(0))  # Average across batch
        self.h = F.sigmoid(h_new.mean(0))
        self.V = torch.tanh(v_new.mean(0)) * (self.v_threshold - self.v_reset) + self.v_reset
        
        # Compute gating function
        g_t = self.beta * (self.m ** 3) * self.h * (self.V - self.v_reset)  # [C]
        
        # Apply gating to input (broadcast across spatial dimensions)
        return x * g_t.view(1, -1, 1, 1)  # [B, C, H, W]