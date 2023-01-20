import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.meta = list(sorted(os.listdir(os.path.join(root, "metadata"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        meta_path = os.path.join(self.root, "metadata", self.meta[idx])
        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(meta_path)
        root = tree.getroot()
        boxes = []
        num_objs = 0
        for child in root:
            if child.tag == "object":
                for grandchild in child:
                    if grandchild.tag == "bndbox":
                        box = {coord.tag: coord.text for coord in grandchild}
                        if 'xmin' in box:
                            num_objs += 1
                            assert int(box['xmin']) < int(box['xmax'])
                            assert int(box['ymin']) < int(box['ymax'])
                            boxes.append([int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)