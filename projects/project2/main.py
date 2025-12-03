import numpy as np
import argparse
import model
from data_loader import *
import os
from skimage import io, util
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import sys 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', help='Load trained model from path', required=False, type=str, default=None)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', required=False, type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', help='Batch size', required=False, type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Number of epochs', required=False, type=int, default=100)
    parser.add_argument('-d', '--depth', help='Depth of UNet Model', required=False, type=int, default=4)
    parser.add_argument('-fls', '--first_layer_size', help='Size of first layer', required=False, type=int, default=64)
    parser.add_argument('-r', '--record', help='Outputs log to .txt file', required=False, action='store_true')

    args = parser.parse_args()

    lr = args.learning_rate
    
    if args.record:
        stdoutOrigin=sys.stdout 
        sys.stdout = open(f"plots/log_lr{lr}_depth{args.depth}_fls{args.first_layer_size}_annotation{1}.txt", "w")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    train_dataset = f"dataset/generated_cells"
    dataset = SegmentationDataset(train_dataset)
    print(f'Load images {train_dataset}: {len(dataset)} ')

    val_ratio = 0.2
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val

    perm = torch.randperm(len(dataset)).tolist()
    train_dataset = Subset(dataset, perm[n_val:])
    val_dataset = Subset(dataset, perm[:n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Number of image in Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    n_classes = 1
    n_channels = 1


    #num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"UNet: {num_params} number of trainable parameters of model ")
    print(f'Pretrained model: {args.load_model}')

    if args.load_model:
        model = model.UNet(0,0, load_from=args.load_model)
    else:
        model = model.UNet(n_channels, args.first_layer_size, n_classes, args.depth)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet: {num_params} number of trainable parameters of model ")


    start_time = time.time()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.4)
    print(f"Using device: {device}")

    model.to(device)
    print(f"Starting training for {args.epochs} epochs...")

    train_losses, val_losses, val_ious = [], [], []
    done_epochs = 0

    for epoch in range(args.epochs):
        done_epochs += 1
        start_epoch_time = time.time()
        model.train()
        running = 0.0
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            msks = msks.unsqueeze(1)
            loss = criterion(model(imgs), msks)
            loss.backward()
            scheduler.step(loss)
            optimizer.step()
            running += loss.item() * imgs.size(0)
        tr_loss = running / len(train_loader)
        train_losses.append(tr_loss)

        model.eval()
        val_running = 0.0
        iou_running = 0.0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device).float()
                msks = msks.unsqueeze(1)
                out = model(imgs)

                vloss = criterion(out, msks)
                val_running += vloss.item() * imgs.size(0)

                pred = torch.sigmoid(out)
                pred = (pred > 0.5).float()

                iou_running += jaccard_score(
                    msks.cpu().numpy().reshape(-1),
                    pred.cpu().numpy().reshape(-1)
                )
        va_loss = val_running / len(val_loader)
        va_iou = iou_running / len(val_loader)
        val_losses.append(va_loss)
        val_ious.append(va_iou)
        print(f"Epoch [{epoch+1}/{args.epochs}]  train_loss: {tr_loss:.4f}  val_loss: {va_loss:.4f} iou:{va_iou:.4f}  time: {time.time() - start_epoch_time:.2f}s lr: {scheduler.get_last_lr()[0]:.2E}")

        # ---- snapshot every N/10 epochs on the first training sample
        """
        if epoch % 10 == 0:
            visualize_prediction(model, train_dataset, device, sample_idx=0)
        """

        if scheduler.get_last_lr()[0] < 1e-8:
            break
            

    print(f"Total time : {time.time() - start_time:2f}s")
    epochs_range = range(1, done_epochs + 1)

    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves")
    plt.legend()

    if args.record:
        plt.savefig(f'plots/losscurves_lr{lr}_depth{args.depth}_fls{args.first_layer_size}_annotation{1}.png')
    else:
        plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, val_ious, label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title(f"IoU Curve")
    plt.legend()

    if args.record:
        plt.savefig(f'plots/ioucurve_lr{lr}_depth{args.depth}_fls{args.first_layer_size}_annotation{1}.png')
    else:
        plt.show()

    if args.record:
        sys.stdout.close()
        sys.stdout=stdoutOrigin
    


