import torch
import torch.nn as nn
import torch.nn.functional as F

class HodgkinHuxleyGating(nn.Module):
    """Implements Hodgkin-Huxley inspired gating mechanism"""
    def __init__(self, channels, beta=0.1, v_threshold=0.5, v_reset=0.0):
        super().__init__()
        self.channels = channels
        self.beta = beta
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        # Initialize gating variables
        self.register_buffer('n', torch.zeros(1, channels, 1, 1))
        self.register_buffer('m', torch.zeros(1, channels, 1, 1))
        self.register_buffer('h', torch.zeros(1, channels, 1, 1))
        
        # Neural network for computing gating variables
        self.gate_network = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels * 3)  # 3 gates: n, m, h
        )
        
    def reset_state(self):
        """Reset gating variables to initial state"""
        device = next(self.parameters()).device
        self.n = torch.zeros(1, self.channels, 1, 1, device=device)
        self.m = torch.zeros(1, self.channels, 1, 1, device=device)
        self.h = torch.zeros(1, self.channels, 1, 1, device=device)
        
    def forward(self, x):
        """
        Forward pass implementing HH-like gating
        Args:
            x: Input tensor [B, C, H, W]
        """
        # Compute channel-wise statistics
        x_stats = torch.mean(x, dim=[2, 3])  # [B, C]
        
        # Get gating variables from neural network
        gate_output = self.gate_network(x_stats)  # [B, C*3]
        B, _ = gate_output.shape
        
        # Split into individual gates
        n_gate, m_gate, h_gate = torch.split(gate_output, self.channels, dim=1)
        
        # Reshape gates to match spatial dimensions
        n_gate = n_gate.view(B, -1, 1, 1)
        m_gate = m_gate.view(B, -1, 1, 1)
        h_gate = h_gate.view(B, -1, 1, 1)
        
        # Create new gating variables for this forward pass
        n_new = torch.sigmoid(n_gate)
        m_new = torch.sigmoid(m_gate)
        h_new = torch.sigmoid(h_gate)
        
        # Update gating variables with temporal dynamics
        # Detach previous states to avoid building up the graph
        n_current = self.n.detach()
        m_current = self.m.detach()
        h_current = self.h.detach()
        
        # Compute new states
        self.n = n_current + self.beta * (n_new - n_current)
        self.m = m_current + self.beta * (m_new - m_current)
        self.h = h_current + self.beta * (h_new - h_current)
        
        # Apply gating function using current states
        gated_output = x * self.n * self.m * self.h
        
        # Reset membrane potential if threshold is reached
        above_threshold = (torch.abs(gated_output) > self.v_threshold).float()
        gated_output = gated_output * (1 - above_threshold) + self.v_reset * above_threshold
        
        return gated_output