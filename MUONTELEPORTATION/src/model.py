import torch
from torch import nn


class SimpleMLP(nn.Module):
    """Two-layer ReLU MLP used for exact function-preserving teleportation checks."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)

    @property
    def layer_names(self):
        return ["hidden", "output"]
