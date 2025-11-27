import numpy as np
import argparse
import model
from data_loader import *




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', help='Load trained model from path', required=False, type=str, default=None)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', required=False, type=float, default=0.001)
    parser.add_argument('-bs', '--batch_size', help='Batch size', required=False, type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Number of epochs', required=False, type=int, default=50)
    parser.add_argument('-d', '--depth', help='Depth of UNet Model', required=False, type=int, default=4)
    parser.add_argument('-fls', '--first_layer_size', help='Size of first layer', required=False, type=int, default=64)

    args = parser.parse_args()

    train_dataset = f"dataset/generated_cells"
    dataset = SegmentationDataset(train_dataset)
    val_ratio = 0.2
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val

    perm = torch.randperm(len(dataset)).tolist()
    train_dataset = Subset(dataset, perm[n_val:])
    val_dataset = Subset(dataset, perm[:n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Number of image in Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    #num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"UNet: {num_params} number of trainable parameters of model ")
    print(f'Pretrained model: {args.load_model}')

    if args.load_model:
        model = model.UNet(0,0, load_from=args.load_model)
    else:
        model = model.UNet(1, args.first_layer_size, 2, args.depth)
        model.fit(train_loader, val_loader, args.epochs, "models\\test", args.learning_rate)

    predictions = model.predict(val_loader)
    print(predictions)