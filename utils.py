import torch
import time
import shutil
import os


class EarlyStopping:
    """Early stopping AND saver !
    """

    def __init__(self, epochs=0, out_path=None):
        self.out_path = out_path
        self.patience = epochs
        self.counter = 0
        self.loss_min = None
        self.early_stop = False
        self.is_best = False
        self.filename = os.path.join(self.out_path, 'model.pt.tar')

    def __call__(self, loss, state, minim):

        #####################
        ## Learning begins ##
        #####################
        if minim:
            sign = 1
        else:
            sign = -1
        if self.loss_min is None:
            self.loss_min = loss
            self.is_best = True

        #####################
        ## No progression ###
        #####################
        elif sign * loss > sign * self.loss_min:
            self.counter += 1
            self.is_best = False
            if self.counter < self.patience:
                print('-- There has been {}/{} epochs without improvement on the validation set. --\n'.format(
                    self.counter, self.patience))
            else:
                self.early_stop = True

        ######################
        ## Learning WIP ######
        ######################
        else:
            self.is_best = True
            self.loss_min = loss
            self.counter = 0
            return self.is_best

        self.save_checkpoint(state)

    def save_checkpoint(self, state):
        torch.save(state, self.filename)
        if self.is_best:
            shutil.copyfile(self.filename, self.filename.replace('.pt.tar', '_best.pt.tar'))
