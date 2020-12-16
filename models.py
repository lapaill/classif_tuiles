from torchvision import models
import numpy as np
import torch
from tensorboardX import SummaryWriter
import os

import Equinet
from utils import EarlyStopping


class Classifier():
    def __init__(self, args):
        self.args = args

        self.model_name = args.model_name
        self.name = args.name
        self.counter = {'epochs': 0, 'batches': 0}
        out_path = os.path.join('results', 'trained_models', args.name)
        log_path = os.path.join('results', 'logs', args.name)
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        self.early_stopping = EarlyStopping(epochs=args.epochs, out_path=out_path)
        self.writer = SummaryWriter(log_path)

        self.device = args.device
        self.pretrained = args.pretrained
        self.num_class = args.num_class
        self.frozen = args.frozen

        self.network = self.get_network()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters())

    def optimize_parameters(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        output_batch = self.forward(input_batch)
        loss = self.criterion(output_batch, target_batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def forward(self, x):
        out = self.network(x)
        return out

    def predict(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            output = self.forward(x)
            proba = torch.nn.functional.softmax(output, dim=1)
            preds = torch.argmax(proba, dim=1)
        return output, np.array(preds.detach().cpu().numpy())

    def get_network(self):
        if self.model_name == "equiwrn":
            network = Equinet.Wide_ResNet(28, 3, 0.3, initial_stride=1, N=12, f=True, r=0, fixparams=False)
            if self.frozen:
                self.freeze_net(network)
            network = network.to(self.device)
            return network
        if self.model_name == "resnet18":
            network, in_features = models.resnet18(pretrained=self.pretrained), 512
        if self.model_name == "resnet50":
            network, in_features = models.resnet50(pretrained=self.pretrained), 2048
        if self.model_name == "equi":
            network, in_features = Equinet.small_wrn(4), 256
        network.fc = torch.nn.Linear(in_features=in_features, out_features=self.num_class)
        if self.frozen:
            self.freeze_net(network)
        network = network.to(self.device)
        return network

    def freeze_net(self, net):
        for name, p in net.named_parameters():
            if 'fc' not in name:
                p.requires_grad = False

    def make_state(self):
        dictio = {'state_dict': self.network.state_dict(),
                  'state_dict_optimizer': self.optimizer.state_dict,
                  'inner_counter': self.counter}
        return dictio
