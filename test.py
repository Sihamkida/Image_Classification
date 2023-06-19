import argparse
import datetime
import torch
import json
import numpy as np

from utils import DatasetCachingWrapper, DatasetIndexingWrapper, ClassificationMetricCalculator, prepare_input
from dataset import Dataset
from model import ConvNet, ResNet18

from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-md', '--model_directory', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int,
                        default=8, required=False)
    parser.add_argument('-nw', '--num_workers', type=int,
                        default=4, required=False)
    parser.add_argument('-s', '--seed', type=int, default=1, required=False)
    parser.add_argument('-c', '--comment', type=str,
                        default='', required=False)

    args = parser.parse_args()
    print(vars(args))
    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()

    print(f'Loading from {args.model_directory}.')
    model_directory = Path(args.model_directory)
    with open(model_directory / 'args.json', "r") as file:
        args_dict = json.load(file)
    training_args = argparse.Namespace(**args_dict)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_test = DatasetCachingWrapper(Dataset(base_directory='dataset', test=True, material_filter='ALU'))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=True)

    with open(model_directory / 'model.json', "r") as file:
        net_args = json.load(file)
    # net = ConvNet(**net_args).to(device)
    net = ResNet18(**net_args).to(device)

    state = torch.load(model_directory / 'state.pkl')
    net.load_state_dict(state['model'])

    logdir = f'tests/{datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "_")}_{model_directory.name}_{args.comment}'

    metric_calculator = ClassificationMetricCalculator()

    net.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        criterion = torch.nn.CrossEntropyLoss()
        batch_losses = []
        for batchindex, batch in enumerate(tqdm(dataloader_test)):
            input, gt = prepare_input(batch, train=False)
            output = net(input)
            loss = criterion(output, gt['grain_size'])
            batch_losses.append(float(loss))
            metric_calculator.add(output, gt['grain_size'])

    test_accuracy = metric_calculator.finish()
    test_loss = np.mean(batch_losses)

    print(f'test loss: {test_loss:.6f}, test accuracy: {test_accuracy:.6f}')
