from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class RandomRotate90(object):
     def __init__(self, choice='uniform'):
         self.choice = choice

     def _get_param(self):
         if self.choice == 'uniform':
             k = np.random.choice([0, 90, 180, 270])
         return k

     def __call__(self, img):
         angle = self._get_param()
         return vision_F.rotate(img, angle)

def get_dataloader(datadir, batch_size, pretrained, augmented):
    dataset = ImageFolder(datadir, transform=get_transforms(pretrained, augmented))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_transforms(pretrained, augmented):
    if pretrained:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.7364, 0.5600, 0.7052],
                                         std=[0.229, 0.1584, 0.1330])
    if augmented:
        print('Use Augmentation plus')
        trans = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRotate90(),
                transforms.ToTensor(),
                normalize
                ])
    else:
        trans = transforms.Compose([
                transforms.ToTensor(),
                normalize
                ])
    return trans
