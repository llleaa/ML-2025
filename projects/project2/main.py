import numpy as np
import argparse
import model




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', help='Load trained model from path', required=False, type=str, default=None)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', required=False, type=float, default=0.001)
    parser.add_argument('-bs', '--batch_size', help='Batch size', required=False, type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Number of epochs', required=False, type=int, default=50)

    args = parser.parse_args()

    if args.load_model:
        model = model.UNet(0,0, load_from=args.load_model)


