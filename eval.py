import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import sklearn.metrics as metrics
import torch
from torchvision import models as models

from dataloader import get_dataloader
from models import Classifier

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--state_dict", type=str,
                    help="Path to the weights file")
parser.add_argument("--test_dir", type=str,
                    help="Path to the test directory")
parser.add_argument("--batch_size", type=int, default=256)


def get_value(tensor):
    return tensor.detach().cpu().numpy()


def make_message(loss, accuracy):
    msg = "Validation results : Loss {} | Accuracy {}".format(
        loss, accuracy)
    return msg


def predict(model, x):
    x = x.to('cpu')
    with torch.no_grad():
        output = model.forward(x)
        proba = torch.nn.functional.softmax(output, dim=1)
        preds = torch.argmax(proba, dim=1)
    return output, np.array(preds.detach().cpu().numpy())


def eval(model, dataloader, loss_F=torch.nn.CrossEntropyLoss()):
    model.eval()
    accuracy = []
    loss = []
    y_pred = []
    y_true = []
    i = 0
    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to('cpu', dtype=torch.int64)
        output, pred = predict(model, input_batch)
        loss.append(loss_F(output, target_batch).item())
        y_pred += list(pred)
        y_true += list(get_value(target_batch))
        i += 1
        print('Batch n°{}, batch size: {} loss for this batch: {}'.format(
            i, len(list(pred)), loss[-1]))
    accuracy = metrics.accuracy_score(y_true, y_pred)
    loss = np.mean(loss)
    msg = make_message(loss, accuracy)
    print(msg)


def main():
    args = parser.parse_args()
    print('loading test dataset')
    test_loader = get_dataloader(
        args.test_dir, args.batch_size, True, False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device: {}'.format(device))

    # loading the model
    model = models.resnet18(
        pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.BatchNorm1d(num_features=512),
        torch.nn.Linear(in_features=512, out_features=512),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=512, out_features=2))

#    print("=> loading model from state dict '{}'".format(args.state_dict))
#    weights_file = torch.load(args.state_dict, map_location="cpu")
#    state_dict = weights_file['state_dict']
#    print(state_dict.keys())
#    model.load_state_dict(state_dict)
#    print('model loaded successfully')

    test_batch = next(iter(test_loader))
    test_input, test_target = test_batch
    test_input = test_input.to(device)
    output = predict(model, test_input)
    test_target = test_target.to('cpu', dtype=torch.int64)
    loss = []
    y_pred = []
    y_true = []
    output, pred = predict(model, test_input)
    loss_f = torch.nn.CrossEntropyLoss()
    loss.append(
        loss_f(output, test_target).item())
    print('loss test: {}'.format(loss))
    y_pred += list(pred)
    y_true += list(get_value(test_target))
    eval(model, test_loader)


if __name__ == "__main__":
    main()
