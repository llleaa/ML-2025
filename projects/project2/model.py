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
        for _ in range(depth):
            self.down_layers.append(model_parts.Down(ch, ch * 2))
            ch *= 2

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
