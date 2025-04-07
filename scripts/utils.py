import torch
from torchvision import transforms
from PIL import ImageDraw
import matplotlib.pyplot as plt

def calculate_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union_area = pred_area + gt_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_map(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp = 0
    fp = 0
    for pred_box, gt_box in zip(pred_boxes, gt_boxes):
        iou = calculate_iou(pred_box, gt_box)
        if iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
    return precision, recall

def visualize_sample(sample):
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    ])
    image_tensor = sample['image']
    denormalized_image = denormalize(image_tensor)
    denormalized_image = torch.clamp(denormalized_image, 0, 1)
    image = transforms.ToPILImage()(denormalized_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, width, height = sample['bbox']
    draw.rectangle([x, y, x + width, y + height], outline="red", width=3)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"Caption: {sample['caption']}")
    plt.show()
