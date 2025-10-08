import torch
import torch.nn as nn

class DrumLogisticRegression(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, num_features)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, T, K)
        output: same shape, probabilities
        """
        out = self.linear(x)
        return self.activation(out)