from model_base import Model
from torchvision import models
import numpy as np
import torch
from tensorboardX import SummaryWriter

import Equinet
from Equinet import wrn28_10_d8d4d1, WideBasic
import e2cnn.nn as enn


class Classifier(Model):
    def __init__(self, args, writer=False):
        super(Classifier, self).__init__(args)
        self.model_name = args.model_name
        self.name = args.name
        self.pretrained = args.pretrained
        self.num_class = args.num_class
        self.frozen = args.frozen
        self.network = self.get_network()
        self.criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.parameters())
        self.optimizers = [optimizer]
        self.get_schedulers()
        self.writer = self.get_writer(writer)

    def optimize_parameters(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        output_batch = self.forward(input_batch)
        self.set_zero_grad()
        loss = self.criterion(output_batch, target_batch)
        loss.backward()
        self.optimizers[0].step()
        return loss

    def get_writer(self, writer):
        if writer:
            writer = SummaryWriter(self.get_folder_writer())
        else:
            writer = None
        return writer

    def forward(self, x):
        out = self.network(x)
        return out

    def predict(self, x):
        x = x.to(self.device)
        output = self.forward(x)
        proba = torch.nn.functional.softmax(output, dim=1)
        preds = torch.argmax(proba, dim=1)
        return output, np.array(preds.detach().cpu().numpy())

    def get_network(self):
        networks = {
            "resnet18": (models.resnet18(pretrained=self.pretrained), 512),
            "resnet50": (models.resnet50(pretrained=self.pretrained), 2048),
            "equi": (Equinet.small_wrn(4), 256)
        }
        network, in_features = networks[self.model_name]
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
                  'state_dict_optimizer': self.optimizers[0].state_dict,
                  'state_scheduler': self.schedulers[0].state_dict(),
                  'inner_counter': self.counter}
        return dictio
