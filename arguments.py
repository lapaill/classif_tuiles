from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int,
                        help="number of epochs to train", default=10)
    parser.add_argument("--datadir", type=str, help="path to the data")

    parser.add_argument("--name", type=str, default="default",
                        help='name of the experiment for logging')
    parser.add_argument("--batch_size", type=int,
                        help="Batch size", default=256)
    parser.add_argument('--model_name', type=str, help='Name of the network to use: resnet18 | resnet50 use perso for SSL MoCo',
                        default='resnet18')
    parser.add_argument('--pretrained', action='store_true',
                        help='If >=1 loads the weights trained on imagenet')
    parser.add_argument('--augmented', action='store_true',
                        help="use augmentations during training ?")
    parser.add_argument('--frozen', action='store_true',
                        help='If >= 1, freeze the network except fc layer')
    parser.add_argument('--weights_file', type=str,
                        help='Weights file for MoCov2 with resnet18')

    return parser
