from torch import nn, optim
import torch
import model_parts
import time

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, load_from=None):

        if load_from is not None:
            self.load_state_dict(torch.load(load_from))
            return

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (model_parts.DoubleConv(n_channels, 64)) #TODO try decreasing
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

    def fit(self, train_dataloader, val_dataloader, max_epochs, save_path, lr=0.001):

        if self.n_classes == 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.to(device)
        train_losses, val_losses = [], []
        for epoch in range(max_epochs):
            start_time = time.time()
            self.train()
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            running = 0
            for X, y in train_dataloader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.forward(X), y)
                loss.backward()
                optimizer.step()
                running += loss.item() * X.size(0)

            train_losses.append(running / max(1, len(train_dataloader)))

            self.eval()
            running = 0
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    loss = criterion(self.forward(X), y)
                    running += loss.item() * X.size(0)
                val_losses.append(running / max(1, len(val_dataloader)))

            print(f"Epoch [{epoch + 1}/{max_epochs}]  train_loss: {train_losses[-1]:.4f}  val_loss: {val_losses[-1]:.4f}  time: {time.time() - start_time:.2f}s")

        self.save_model(save_path)

        return train_losses, val_losses


    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X)



    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
