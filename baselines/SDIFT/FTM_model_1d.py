import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)

    def forward(self, x):
        return torch.sin(torch.sin(self.omega_0 * self.linear(x)))


class Tensor_inr_1D(nn.Module):
    """Continuous spatial basis for 1D fields.

    Forward returns U of shape (X, R1) where X is the number of supplied
    spatial coordinates and R1 is the spatial rank.
    """

    def __init__(self, R1: int, omega: float = 20.0, mid_channel: int = 1024):
        super().__init__()
        self.r_1 = R1
        self._mode = "training"
        self.U_net = nn.Sequential(
            SineLayer(1, mid_channel, omega_0=omega),
            SineLayer(mid_channel, mid_channel, omega_0=omega),
            nn.Dropout(0.0),
            nn.Linear(mid_channel, R1),
            nn.Tanh(),
        )

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        if m not in ("training", "sampling"):
            raise ValueError("mode must be 'training' or 'sampling'")
        self._mode = m

    def forward(self, x_ind):
        # x_ind: (X,) or (X, 1) tensor of normalized spatial positions in [0, 1].
        if x_ind.dim() == 1:
            x_ind = x_ind.unsqueeze(1)
        return self.U_net(x_ind)  # (X, R1)
