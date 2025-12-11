import numpy as np
import argparse
import model as md
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
    parser.add_argument('-s', '--sample', help='Number of training samples', required=False, type=int, default=500)
    parser.add_argument('-f', '--fold', help='Number of folds that the model is trained', required=False, type=int, default=3)
    parser.add_argument('-a', '--annotation', help='Annotation quality [1,0.9,0.78,0.72,0.57]', required=False, type=float, default=1)


    args = parser.parse_args()

    lr = args.learning_rate

    if not args.annotation in [1,0.9,0.78,0.72,0.57]:
        raise Exception('bad annotation value, see usage')
    
    if args.annotation == 1:
        train_dataset_path = f"dataset/generated_cells"
    else:
        train_dataset_path = f"dataset/eroded-dilated_{args.annotation}"
    
    val_dataset_path = f"dataset/generated_cells"

    train_dataset = SegmentationDataset(train_dataset_path)

    val_dataset = SegmentationDataset(val_dataset_path)

    #val_dataset = SegmentationDataset(f"dataset/eroded-dilated_{args.annotation}")
    img, msk = val_dataset[4]

    #print(msk.max(), msk.min(), np.count_nonzero(msk))


    print("my image : ", np.count_nonzero(val_dataset[40][1]))
    print("my image : ", np.count_nonzero(train_dataset[40][1]))

    
    val_ratio = 0.2
    n_train = int(min(args.sample, len(train_dataset) * (1-val_ratio)))
    n_val = len(train_dataset) - n_train


    if args.record:
        stdoutOrigin=sys.stdout 
        sys.stdout = open(f"plots/log_depth{args.depth}_fls{args.first_layer_size}_annotation{1}_sample{n_train}.txt", "w")
        df = pd.read_csv('results.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f'Load images {train_dataset_path}: {len(train_dataset)} ')

    mean_time = 0
    mean_iou = 0

    for fold in range(args.fold):

        perm = torch.randperm(len(train_dataset)).tolist()
        print(perm)
        train_dataset = Subset(train_dataset, perm[n_val:])
        val_dataset = Subset(val_dataset, perm[:n_val])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        print(f"Number of image in Train: {len(train_dataset)} | Val: {len(val_dataset)}")

        n_classes = 1
        n_channels = 1

        print(f'Pretrained model: {args.load_model}')

        if args.load_model:
            model = md.UNet(0,0, load_from=args.load_model)
        else:
            model = md.UNet(n_channels, args.first_layer_size, n_classes, args.depth)

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

                    pred = (torch.sigmoid(out) > 0.5).float()

                    intersection = (pred * msks).sum()
                    union = pred.sum() + msks.sum() - intersection

                    iou = (intersection / (union + 1e-6)).item()
                    iou_running += iou

                    #iou_running += jaccard_score(
                    
                        #msks.squeeze(1).cpu().numpy().reshape(-1).astype(int),
                        #pred.squeeze(1).cpu().numpy().reshape(-1).astype(int),
                        #average="binary"
                    #)
            va_loss = val_running / len(val_loader)
            va_iou = iou_running / len(val_loader)
            val_losses.append(va_loss)
            val_ious.append(va_iou)
            print(f"Epoch [{epoch+1}/{args.epochs}]  train_loss: {tr_loss:.4f}  val_loss: {va_loss:.4f} iou:{va_iou:.4f}  time: {time.time() - start_epoch_time:.2f}s lr: {scheduler.get_last_lr()[0]:.2E}")

            if scheduler.get_last_lr()[0] < 1e-7:
                break

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        t = time.time() - start_time


        epochs_range = range(1, done_epochs + 1)
        plt.figure(figsize=(10,5))
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves")
        plt.legend()

        if args.record:
            plt.savefig(f'plots/losscurves_depth{args.depth}_fls{args.first_layer_size}_annotation{1}_sample{n_train}_fold{fold}.png')
        else:
            plt.show()

        plt.figure(figsize=(10,5))
        plt.plot(epochs_range, val_ious, label="Val IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title(f"IoU Curve")
        plt.legend()

        if args.record:
            plt.savefig(f'plots/ioucurve_depth{args.depth}_fls{args.first_layer_size}_annotation{1}_sample{n_train}_fold{fold}.png')
        else:
            plt.show()


        mean_time += t
        mean_iou += val_ious[-1]

    mean_time /= args.fold
    mean_iou /= args.fold

    print(f"Mean time : {mean_time:2f}s")

    if args.record:
        df.loc[len(df)] = [int(args.depth), int(args.first_layer_size), 1, int(n_train), mean_iou, mean_time, int(num_params)]
        df.to_csv('results.csv', index=False)

    if args.record:
        sys.stdout.close()
        sys.stdout=stdoutOrigin
    


