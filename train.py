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
    parser.add_argument('-e', '--epochs', type=int,
                        default=200, required=False)
    parser.add_argument('-b', '--batch_size', type=int,
                        default=8, required=False)
    parser.add_argument('-lr', '--learning_rate',
                        type=float, default=1e-4, required=False)
    parser.add_argument('-m', '--momentum', type=float,
                        default=0.9, required=False)
    parser.add_argument('-w', '--weight_decay', type=float,
                        default=1e-4, required=False)
    parser.add_argument('-nw', '--num_workers', type=int,
                        default=4, required=False)
    parser.add_argument('-s', '--seed', type=int, default=2, required=False)

    parser.add_argument('-lrss', '--lr_scheduler_step_size',
                        type=float, default=30, required=False)
    parser.add_argument('-lrsf', '--lr_scheduler_step_factor',
                        type=float, default=0.5, required=False)

    parser.add_argument('-save', '--save_frequency',
                        type=int, default=1, required=False)

    parser.add_argument('-o', '--overfit', type=bool,
                        default=False, required=False)
    parser.add_argument('-c', '--comment', type=str,
                        default='', required=False)
    parser.add_argument('-rd', '--resume_directory', type=str,
                        default=None, required=False)

    args = parser.parse_args()
    print(vars(args))
    return args


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()

    resume_directory = None
    if args.resume_directory is not None:
        epochs = args.epochs
        print(f'Resuming from {args.resume_directory}.')
        resume_directory = Path(args.resume_directory)
        with open(resume_directory / 'args.json', "r") as file:
            args_dict = json.load(file)
        args = argparse.Namespace(**args_dict)
        args.epochs = epochs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = DatasetCachingWrapper(Dataset(base_directory='dataset', material_filter='ALU'))
    indices_train, indices_val = train_test_split(
        range(len(dataset)), test_size=0.2, shuffle=True, random_state=args.seed)
    dataset_train = DatasetIndexingWrapper(dataset, indices_train)
    dataset_val = DatasetIndexingWrapper(dataset, indices_val)

    if args.overfit:
        args.comment += '_overfit'
        args.weight_decay = 0.0
        dataset_train = DatasetIndexingWrapper(dataset_train, [0])
        dataset_val = DatasetIndexingWrapper(dataset_train, [0])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,  num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    if resume_directory is not None:
        with open(resume_directory / 'model.json', "r") as file:
            net_args = json.load(file)
    else:
        # net_args = dict(input_dim=3, input_size=256, output_dim=7, channel_sizes=[16, 32, 64, 128])
        net_args = dict(output_dim=7)
    # net = ConvNet(**net_args).to(device)
    net = ResNet18(**net_args).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        betas=(args.momentum, 0.999),
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_scheduler_step_size,
        gamma=args.lr_scheduler_step_factor
    )

    start_epoch = 0

    if resume_directory is not None:
        state = torch.load(resume_directory / 'state.pkl')
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch']

    logdir = f'logs/{datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "_")}_{args.comment}'
    writer = SummaryWriter(log_dir=logdir)
    logdir = Path(writer.log_dir)
    writer.add_text('args', json.dumps(vars(args), indent=4))
    writer.add_text('model', json.dumps(net_args, indent=4))
    with open(logdir / 'args.json', "w") as file:
        json.dump(vars(args), file, indent=4)
    with open(logdir / 'model.json', "w") as file:
        json.dump(net_args, file, indent=4)

    metric_calculator = ClassificationMetricCalculator()

    writer.add_custom_scalars({
        "_combined": {
            k: ['Multiline', [f'{mode}/{k}' for mode in ['train', 'val']]] for k in ['loss_epoch', 'accuracy_epoch']
        }
    })

    for epoch in range(start_epoch, args.epochs):
        def loop(train=True):
            torch.cuda.empty_cache()
            criterion = torch.nn.CrossEntropyLoss()
            batch_losses = []
            dataloader = dataloader_train if train else dataloader_val
            for batchindex, batch in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                input, gt = prepare_input(batch, train=train)
                output = net(input)
                num_batch = epoch * len(dataloader) + batchindex
                loss = criterion(output, gt['grain_size'])
                if train:
                    loss.backward()
                    optimizer.step()
                torch.cuda.empty_cache()
                batch_losses.append(float(loss))
                writer.add_scalar(
                    f'{"train" if train else "val"}/loss_batch', float(loss), num_batch)
                with torch.no_grad():
                    metric_calculator.add(output, gt['grain_size'])

            with torch.no_grad():
                accuracy = metric_calculator.finish()
            return np.mean(batch_losses), accuracy

        net.train()
        train_loss, train_accuracy = loop(train=True)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        net.eval()
        with torch.no_grad():
            val_loss, val_accuracy = loop(train=False)
            if (epoch % args.save_frequency == 0 or epoch == args.epochs - 1):
                state = dict(
                    epoch=epoch + 1,
                    model=net.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                )
            torch.save(state, logdir / 'state.pkl')

            writer.add_scalar(f'train/loss_epoch', float(train_loss), epoch)
            writer.add_scalar(f'val/loss_epoch', float(val_loss), epoch)
            writer.add_scalar(f'train/accuracy_epoch',
                              float(train_accuracy), epoch)
            writer.add_scalar(f'val/accuracy_epoch',
                              float(val_accuracy), epoch)
            writer.flush()
            print(
                f'Epoch: {epoch=:03}, train loss: {train_loss:.6f}, train accuracy: {train_accuracy:.6f}')
            print(
                f'Epoch: {epoch=:03}, val loss: {val_loss:.6f}, val accuracy: {val_accuracy:.6f}')
            scheduler.step()
