"""A small configurable multilayer perceptron."""

import torch
from torch import nn


class Layer(nn.Module):
    """A linear layer with an optional ReLU activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_activation_fn: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if has_activation_fn:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x


class MLP(nn.Module):
    """An MLP with configurable hidden-layer sizes."""

    def __init__(
        self,
        hidden_sizes: list[int] | None,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or []
        sizes = [input_size, *hidden_sizes, output_size]
        layers = [
            Layer(
                in_features=sizes[index],
                out_features=sizes[index + 1],
                has_activation_fn=index < len(sizes) - 2,
                bias=bias,
            )
            for index in range(len(sizes) - 1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
