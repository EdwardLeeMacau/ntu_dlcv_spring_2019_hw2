import argparse
import os

from pkg_resources import require

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import dataset
from dataset import Testset
import models
import utils

classnames = dataset.classnames
labelEncoder  = dataset.labelEncoder
oneHotEncoder = dataset.oneHotEncoder

def decode(output: torch.Tensor, nms=True, prob_min=0.1, iou_threshold=0.5, grid_num=7, bbox_num=2, class_num=16):
    """
    Parameters
    ----------
    output : torch.Tensor
        shape: [batch_size, grid_num, grid_num, 5 * bbox_num + class_num]

    nms : bool
        Open nms option

    prob_min : float

    iou_threshold : float

    grid_num : int

    class_num : int

    Return
    -------
    keep_boxes: <list of bbox>

    classNames: <list of list>

    """
    boxes, classIndexs, probs = [], [], []
    cell_size   = 1. / grid_num
    batch_size  = output.shape[0]

    output = output.data
    output = output.squeeze(0) # [7, 7, 26]
    # print("Output.shape: {}".format(output.shape))
    # print("Output: {}".format(output))

    contain1 = output[:, :, 4].unsqueeze(-1)
    contain2 = output[:, :, 9].unsqueeze(-1)
    # print("Contain1.shape: {}".format(contain1.shape))
    contain = torch.cat((contain1, contain2), -1)
    # print("Contain.shape: {}".format(contain.shape))
    # print(contain[3, 3])

    mask1 = (contain > prob_min)
    mask2 = (contain == contain.max())
    mask  = (mask1 + mask2).gt(0)
    # print("Conf.max: {}".format(contain.max().item()))
    # print("Contain[mask]: {}".format(contain[mask]))

    # TODO: parallel processing index i and j
    # i: Row message
    # j: Column message
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(bbox_num):
                if mask[i, j, b] == 1:
                    box = output[i, j, b*5: b*5+4].type(torch.float)
                    contain_prob = output[i, j, b*5+4].type(torch.float)

                    # Recover the base of xy as image_size
                    # xy = torch.tensor([j, i], dtype=torch.float).unsqueeze(0) * cell_size
                    xy = torch.tensor([j, i], dtype=torch.float).cuda().unsqueeze(0) * cell_size

                    box[:2] = box[:2] * cell_size + xy
                    box_xy  = torch.zeros(box.size(), dtype=torch.float)
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, class_idx = torch.max(output[i, j, 10:], 0)

                    # print("Max_Prob: {}".format(max_prob))
                    # print("Contain_prob: {}".format(contain_prob))
                    # pdb.set_trace()

                    if float((contain_prob * max_prob).item()) > prob_min:
                        class_idx = class_idx.unsqueeze(0)
                        boxes.append(box_xy.view(1, 4))
                        classIndexs.append(class_idx)
                        probs.append((contain_prob * max_prob).view(1))

    # TODO: define output formats for NULL bbox
    if len(boxes) == 0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        classIndexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0) #(n,4)
        probs = torch.cat(probs, 0) #(n,)
        classIndexs = torch.cat(classIndexs, 0) #(n,)

    # Prevent the boxes go outside the image, so clamped the xy coordinate to 0-1
    boxes = boxes.clamp(min=0., max=1.)

    if nms:
        keep_index = nonMaximumSupression(boxes, probs, iou_threshold)
        return boxes[keep_index], classIndexs[keep_index], probs[keep_index]

    return boxes, classIndexs, probs

def nonMaximumSupression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold):
    """
    Not generalize to multi-img processing, only 1 in 1 out.

    Parameters
    ----------
    boxes : torch.Tensor
        [N, 4], (x1, y1, x2, y2)
    scores : torch.Tensor
        [N]
    iou_threshold : float
        [1]

    Return
    ------
    keep_boxes :
        [x]
    """
    _, index = scores.sort(descending=True)
    keep_boxes = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    while index.numel() > 0:
        if index.numel() == 1:
            keep_boxes.append(index.item())
            break

        i = index[0].item()
        keep_boxes.append(i)

        # IoU calculating
        xx1 = x1[index[1:]].clamp(min=x1[i])
        yy1 = y1[index[1:]].clamp(min=y1[i])
        xx2 = x2[index[1:]].clamp(max=x2[i])
        yy2 = y2[index[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[index[1:]] - inter)

        # Supress the bbox where overlap area > iou_threshold, return the remain index
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        # IoU calculated.

        if ids.numel() == 0: break
        index = index[ids + 1]

    return torch.tensor(keep_boxes, dtype=torch.long)

def export(boxes, classNames, probs, label_name, out_path, image_size=512.):
    """
    Write textfile with the boxes and the classnames.

    Parameters
    ----------
    boxes :

    classNames :

    probs :

    label_name : str

    out_path : str
        The directory of the generated text file.

    image_size : float
        The original size of the image.
    """
    boxes = (boxes * image_size).round()
    rect  = torch.zeros(boxes.shape[0], 8)

    # Extand (x1, y1, x2, y2) to (x1, y1, x2, y1, x2, y2, x1, y2)
    rect[:,  :3] = boxes[:, :3]
    rect[:, 3:6] = boxes[:, 1:]
    rect[:, 6]   = boxes[:, 0]
    rect[:, 7]   = boxes[:, 3]

    # Return the probs to string lists
    probs = list(map(str, list(map(lambda x: round(x, 3), probs.data.tolist()))))
    classNames = list(map(str, classNames))

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, os.path.basename(label_name)), "w") as textfile:
        for i in range(0, rect.shape[0]):
            prob = probs[i]
            className = classNames[i]

            textfile.write(" ".join(map(str, rect[i].data.tolist())) + " ")
            textfile.write(" ".join((className, prob)) + "\n")

def main(args):
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    torch.set_default_dtype(torch.float)
    device = utils.selectDevice()

    # Initialize model
    if args.command == "basic":
        model = models.Yolov1_vgg16bn(pretrained=True).to(device)
    elif args.command == "improve":
        model = models.Yolov1_vgg16bn_Improve(pretrained=True).to(device)

    model = utils.loadModel(args.model, model)
    model.eval()

    # Initialize dataset
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(
        # TODO: Reset as AerialDataset
        Testset(img_root=args.images, transform=transform),
        batch_size=1, shuffle=False, num_workers=4
    )

    # Iterative interference
    for data, imgName in tqdm(loader):
        data = data.to(device)
        output = model(data)

        if args.command == "basic":
            boxes, class_idxs, probs = decode(output, nms=args.nms, prob_min=args.prob, iou_threshold=args.iou)
        if args.command == "improve":
            boxes, class_idxs, probs = decode(output, nms=args.nms, prob_min=args.prob, iou_threshold=args.iou, grid_num=14)

        classNames = labelEncoder.inverse_transform(class_idxs.type(torch.long).to("cpu"))

        export(boxes, classNames, probs, imgName[0] + ".txt", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Load model parameter")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--nms", action="store_true", help="Open NMS")
    parser.add_argument("--iou", default=0.5, type=float, help="NMS iou_threshold")
    parser.add_argument("--prob", default=0.1, type=float, help="NMS prob_min, pick up the bbox with the class_prob > prob_min")

    subparsers = parser.add_subparsers(required=True, dest="command")
    basic_parser = subparsers.add_parser("basic")
    improve_parser = subparsers.add_parser("improve")

    args = parser.parse_args()

    main(args)
