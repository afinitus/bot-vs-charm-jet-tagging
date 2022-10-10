import torch
import torch.nn as nn


class GlobalAttentionPooling(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = nn.Linear(input_size, 1)

    def forward(self, x, mask, get_attention=False):
        weights = torch.softmax(self.gate_nn(x), dim=1)
        weights[mask] == 0
        weighted = x * weights
        readout = weighted.sum(dim=1)

        if get_attention:
            return readout, weights
        else:
            return readout
