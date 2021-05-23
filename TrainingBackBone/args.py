import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=1, type=int, help="Which GPU to train.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size to use during training.")
    parser.add_argument("--size", default=512, type=int, help="model1 input size")
    parser.add_argument("--datamore", default=1, type=int, help="DataAugmentation")
    parser.add_argument("--optim", default=0, type=int, help="Optimizer: 0: Adam, 1: SGD, 2:SGD with Nesterov")
    parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
    parser.add_argument("--lr1", default=0.00005, type=float, help="Learning rate for optimizer")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train")
    parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
    args = parser.parse_args()
    return args
