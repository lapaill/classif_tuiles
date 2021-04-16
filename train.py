import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Local imports
from arguments import get_parser
from dataloader import get_dataloader
from models import Classifier


def make_message(loss, accuracy, epoch, is_best):
    msg = "Validation results, EPOCH {} : Loss {} | Accuracy {}".format(
        epoch, loss, accuracy)
    if is_best:
        msg = msg.upper()
    return msg


def train(model, dataloader):
    model.network.train()
    for input_batch, target_batch in tqdm(dataloader):
        model.counter['batches'] += 1
        loss = model.optimize_parameters(input_batch, target_batch)
        if model.writer:
            model.writer.add_scalar(
                "Training_loss", loss, model.counter['batches'])
    model.counter['epochs'] += 1


def val(model, dataloader):
    model.network.eval()
    accuracy = []
    loss = []
    y_pred = []
    y_true = []
    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to(model.device, dtype=torch.int64)
        output, pred = model.predict(input_batch)
        loss.append(model.criterion(output, target_batch).item())
        y_pred += list(pred)
        y_true += list(get_value(target_batch))
    accuracy = metrics.accuracy_score(y_true, y_pred)
    loss = np.mean(loss)
    state = model.make_state()
    if model.writer:
        model.writer.add_scalar("Validation_loss", loss,
                                model.counter['epochs'])
        model.writer.add_scalar(
            "Validation_acc", accuracy, model.counter['epochs'])
    is_best = model.early_stopping(accuracy, state, minim=False)
    msg = make_message(loss, accuracy, model.counter['epochs'], is_best)
    print(msg)
    with open(os.path.join(model.early_stopping.out_path, 'learning.log'), 'a') as f:
        f.write(msg + "\n")


def get_value(tensor):
    return tensor.detach().cpu().numpy()


def main():
    args = get_parser().parse_args()
    print(args)
    # Make datasets
    train_dir = os.path.join(args.datadir, 'train')
    val_dir = os.path.join(args.datadir, 'val')
    train_loader = get_dataloader(
        train_dir, args.batch_size, args.pretrained, args.augmented)
    val_loader = get_dataloader(
        val_dir, args.batch_size, args.pretrained, False)

    args.num_class = len(train_loader.dataset.classes)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialisation model
    model = Classifier(args=args)

    while model.counter['epochs'] < args.epochs:
        train(model=model, dataloader=train_loader)
        val(model=model, dataloader=val_loader)
        if model.early_stopping.early_stop:
            break
    if model.writer:
        model.writer.close()


if __name__ == "__main__":
    main()
