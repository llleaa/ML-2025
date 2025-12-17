from torch import nn, optim
import torch
import model_parts
import time


class UNet(nn.Module):
    def __init__(self, n_channels, first_layer_size, n_classes, depth, load_from=None):

        if load_from is not None:
            self.load_state_dict(torch.load(load_from))
            return

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.first_layer_size = first_layer_size
        self.n_classes = n_classes
        self.depth = depth

        self.inc = model_parts.DoubleConv(n_channels, first_layer_size)

        self.down_layers = nn.ModuleList()
        ch = first_layer_size

        # Down part of the model
        for _ in range(depth):
            self.down_layers.append(model_parts.Down(ch, ch * 2))
            ch *= 2

        # Up part of the model
        self.up_layers = nn.ModuleList()
        for _ in range(depth):
            self.up_layers.append(model_parts.Up(ch, ch // 2))
            ch //= 2

        self.outc = model_parts.OutConv(first_layer_size, n_classes)


    def forward(self, x):
        skips = []

        # Encoder
        x = self.inc(x)
        skips.append(x)

        for down in self.down_layers:
            x = down(x)
            skips.append(x)

        # Remove bottom layer from skip list and reverse order
        skips = skips[:-1][::-1]

        # Decoder
        for i, up in enumerate(self.up_layers):
            x = up(x, skips[i])

        return self.outc(x)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
