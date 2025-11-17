from torch import nn, optim
import torch
import model_parts

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (model_parts.DoubleConv(n_channels, 64))
        self.down1 = (model_parts.Down(64, 128))
        self.down2 = (model_parts.Down(128, 256))
        self.down3 = (model_parts.Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (model_parts.Down(512, 1024 // factor))
        self.up1 = (model_parts.Up(1024, 512 // factor, bilinear))
        self.up2 = (model_parts.Up(512, 256 // factor, bilinear))
        self.up3 = (model_parts.Up(256, 128 // factor, bilinear))
        self.up4 = (model_parts.Up(128, 64, bilinear))
        self.outc = (model_parts.OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def fit(self, X, y, max_epochs, batch_size, lr=0.001):
        if self.n_classes == 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.to(device)

        for epoch in range(max_epochs):
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            # TODO : finish fit method


        return 0


