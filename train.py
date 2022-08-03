import argparse
import json
import os
from pprint import pprint

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from evaluation import scan_map
import models
import utils
from dataset import labelEncoder
from inference import decode, export


# TODO: Train Backbone / Train the Dense Layer only.
def train(model, criterion, optimizer, scheduler, train_loader, val_loader,
          start_epochs, epochs, device, grid_num=7, save_name="Yolov1"):
    model.train()

    epochs_list     = []
    train_loss_list = []
    val_loss_list   = []
    val_mean_aps    = []
    val_class_aps   = []

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints', exist_ok=True)

    for epoch in range(start_epochs + 1, epochs + 1):
        model.train()

        iteration = 0
        train_loss = 0

        # Train and backpropagation
        with tqdm(train_loader, postfix="Loss: {loss: .6f}") as loop:
            for (data, target, _) in loop:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                iteration += 1

                loop.set_postfix({'loss': loss.item()})

        scheduler.step()

        train_loss /= iteration
        val_loss = test_loss(model, criterion, val_loader, device)

        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print("*** Train set - Average loss: {:.4f}".format(train_loss))
        print("*** Test set - Average loss: {:.4f}".format(val_loss))

        utils.saveCheckpoint(os.path.join('./checkpoints', save_name + "-{}.pth".format(epoch)), model, optimizer, scheduler, epoch)

        class_aps, mean_ap = test_map(model, criterion, val_loader, device, grid_num=grid_num)
        val_class_aps.append(class_aps)
        val_mean_aps.append(mean_ap)

        metrics = {'train_loss': train_loss_list, 'val_loss': val_loss_list, 'AP': val_class_aps, 'mAP': val_mean_aps}
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)

    return model, metrics

@torch.no_grad()
def test_loss(model, criterion, dataloader: DataLoader, device):
    model.eval()
    loss = 0

    for data, target, _ in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion(output, target).item()

    loss /= len(dataloader.dataset)

    return loss

@torch.no_grad()
def test_map(model, criterion, dataloader: DataLoader, device, grid_num: int):
    """
    Evaluate model performance by mAP (mean Average Percision)

    Parameters
    ----------
    model : torch.nn.Module

    loader : torch.utils.data.DataLoader

    grid_num :

    Returns
    -------
    mAP : float
        mAP score
    """
    model.eval()
    mean_ap = 0

    # Calculate the map value
    for data, target, labelNames in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        boxes, class_idx, probs = decode(output, prob_min=args.prob, iou_threshold=args.iou, grid_num=grid_num, bbox_num=2)
        classNames = labelEncoder.inverse_transform(class_idx.type(torch.long).to("cpu"))
        export(boxes, classNames, probs, labelNames[0], out_path="output/labelTxt_hbb_pred")

    classaps, mean_ap = scan_map(
        detpath="output/labelTxt_hbb_pred/",
        annopath="hw2_train_val/val1500/labelTxt_hbb/"
    )

    return classaps, mean_ap

def main(args: argparse.Namespace):
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ColorJitter(brightness=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    grid_num = 7 if args.command == "basic" else 14

    trainset = dataset.AerialDataset(
        root="hw2_train_val/train15000",
        grid_num=grid_num,
        train=args.augmentation,
        transform=transform
    )

    testset  = dataset.AerialDataset(
        grid_num=grid_num,
        root="hw2_train_val/val1500",
        train=False,
        transform=transform
    )

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.workers)
    device = utils.selectDevice()

    if args.command == "basic":
        model = models.Yolov1_vgg16bn(pretrained=True).to(device)
        criterion = models.YoloLoss(7., 2., args.lambda_coord, args.lambda_noobj, device).to(device)
        save_name = "Yolov1"
    elif args.command == "improve":
        model = models.Yolov1_vgg16bn_Improve(pretrained=True).to(device)
        criterion = models.YoloLoss(14., 2., args.lambda_coord, args.lambda_noobj, device).to(device)
        save_name = "Yolov1-Improve"

    start_epoch = 0
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.SGD(model.yolo.parameters(), **args.optimizer['params'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **args.scheduler['params'], verbose=True)

    if args.load:
        model, optimizer, start_epoch, scheduler = utils.loadCheckpoint(args.load, model, optimizer, scheduler)

    model, _ = train(model, criterion, optimizer, scheduler, train_loader, test_loader, start_epoch, args.epochs, device, save_name=save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, help="Reload the checkpoint")

    args = parser.parse_args()

    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for category in ('environment', 'loss', 'model', 'train'):
        for key, values in config[category].items():
            setattr(args, key, values)

    args.command = 'basic' if args.grid == 7 else 'improve'

    pprint(args)

    main(args)
