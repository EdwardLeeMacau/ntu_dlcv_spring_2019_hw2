# Script modify from https://gist.github.com/Amir22010/a99f18ca19112bc7db0872a36a03a1ec

import argparse
from email.mime import base
import glob
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm

classes = ['person', 'car']

def getImagesInDir(dir_path):
    image_list = []

    for filename in glob.glob(dir_path + '/*.xml'):
        image_list.append(filename)

    return image_list

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(dir_path, output_path):
    basename = os.path.splitext(os.path.basename(dir_path))[0]

    with open(os.path.join(dir_path), 'r') as f:
        tree = ET.parse(f)

    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(os.path.join(output_path, basename + '.txt'), 'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def main(args: argparse.Namespace):
    annotation_dirs = getImagesInDir(args.input)
    assert len(annotation_dirs)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # TODO: Consider parallel working...
    with open(os.path.join(args.output, 'summary.txt'), 'w') as f:
        for annotation in tqdm(annotation_dirs):
            # f.write(annotation + '\n')
            convert_annotation(annotation, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to transform VOC annotation to YOLO annotation.")
    parser.add_argument('--input', required=True, help="Input annotation directory.")
    parser.add_argument('--output', required=True, help="Output annotation directory.")
    args = parser.parse_args()

    main(args)
