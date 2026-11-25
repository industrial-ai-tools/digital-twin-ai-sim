import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D Fourier spectral convolution layer."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        x = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1), norm="ortho")
        return x


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return nn.functional.gelu(self.spectral(x) + self.local(x))


class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32, n_blocks=4):
        super().__init__()
        self.fc0 = nn.Linear(2, width)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes1, modes2) for _ in range(n_blocks)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        B, _, H, W = x.shape
        gridx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W)
        gridy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
        grid = torch.cat([
            gridx.expand(B, 1, H, W),
            gridy.expand(B, 1, H, W),
        ], dim=1)

        x = torch.cat([x, grid], dim=1)          # (B, 2, H, W)
        x = x.permute(0, 2, 3, 1)                  # (B, H, W, 2)
        x = self.fc0(x)                            # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)                  # (B, width, H, W)

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)                  # (B, H, W, width)
        x = nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)                            # (B, H, W, 1)
        x = x.permute(0, 3, 1, 2)                  # (B, 1, H, W)
        return x