"""
Segment images of post its by finetuning segmentation model
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torch
from segmentdataset import SegmentationDataset
from transforms import ToTensor, RandomHorizontalFlip, Compose
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import os

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def make_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) 
    return model


def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset('data', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    model = make_model()
    visualise(model=model, path="base_model")  # To assess performance before fine-tuning
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=params, lr=1e-4)
    num_epochs = 12
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for images, targets in data_loader:
            optimizer.zero_grad()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print(losses.item())
            losses.backward()
            optimizer.step()
    torch.save(model, 'finetuned.pt')
    visualise(model=model)  # To assess performance after fine-tuning

    
def visualise(model, dir="validation", path="results"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(dir, get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.to(device)
    model.eval()
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        pred = model(images)
        fig = plt.figure(i, figsize=(20,10))
        ax = fig.add_subplot(1, 2, 1)
        plot_bb(image=images[0], target=pred[0], ax=ax)
        ax.set_title("Predicted")
        ax = fig.add_subplot(1, 2, 2)
        plot_bb(image=images[0], target=targets[0], ax=ax)
        ax.set_title("Actual")
        fig.savefig(os.path.join(path, f"result_{i}.png"))

def plot_bb(image, target, ax):
    boxes = target['boxes']
    boxes = boxes.cpu().to(torch.int32)
    boxes[:, 0] = torch.min(boxes[:, 0], boxes[:, 2] - 1)
    boxes[:, 1] = torch.min(boxes[:, 1], boxes[:, 3] - 1)
    img = (image.cpu() * 255).to(dtype = torch.uint8)
    bb = draw_bounding_boxes(img, boxes, colors="blue", width=6)
    ax.imshow(bb.permute(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    main()