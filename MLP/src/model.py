import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, num_classes: int = 10):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.hidden_dims = list(hidden_dims)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
