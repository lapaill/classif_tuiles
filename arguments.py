from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs",  type=int, help="number of epochs to train", default=10)
    parser.add_argument("--datadir", type=str, help="path to the data")

    parser.add_argument("--out_path", type=str, default=None, help='path to write the outputs')
    parser.add_argument("--batch_size", type=int, help="Batch size", default=256)
    parser.add_argument('--model_name', type=str, help='Name of the network to use: resnet18 | resnet50', default='resnet18')
    parser.add_argument('--pretrained', type=int, help='If >=1 loads the weights trained on imagenet', default=0)
    parser.add_argument('--frozen', type=int, help='If >= 1, freeze the network except fc layer', default=0)
    return parser
