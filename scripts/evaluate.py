import os
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer

from dataset import RefCOCODataset
from utils import calculate_iou, calculate_map


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Visual/Referring Grounding Model")
    parser.add_argument('--config', type=str, help='Path to config JSON file')

    parser.add_argument('--model_type', type=str, default='referring', choices=['referring', 'visual'],
                        help="Choose the model: 'referring' or 'visual'")
    parser.add_argument('--model_path', type=str, default='models/visual_grounding_model.pth')
    parser.add_argument('--image_dir', type=str, default='data/val2014')
    parser.add_argument('--annotation_file', type=str, default='data/instances_val.json')
    parser.add_argument('--refexp_file', type=str, default='data/refs(unc)_val.p')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_cuda', action='store_true')
    return parser.parse_args()


def load_config(args):
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)
    return args


def get_model(model_type):
    if model_type == 'referring':
        from model_referring_grounding import ReferringGroundingModel
        return ReferringGroundingModel()
    elif model_type == 'visual':
        from model_visual_grounding import VisualGroundingModel
        return VisualGroundingModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def evaluate(model, dataloader, tokenizer, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_pred_boxes, all_gt_boxes = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            captions = batch['caption']

            encoding = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            predicted_bboxes, _ = model(images, input_ids, attention_mask)
            loss = loss_fn(predicted_bboxes, bboxes)
            total_loss += loss.item()

            all_pred_boxes.extend(predicted_bboxes.cpu())
            all_gt_boxes.extend(bboxes.cpu())

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    precisions, recalls, ious = [], [], []

    for pred_box, gt_box in zip(all_pred_boxes, all_gt_boxes):
        iou = calculate_iou(pred_box, gt_box)
        precision, recall = calculate_map([pred_box], [gt_box])
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)

    avg_iou = sum(ious) / len(ious)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    return avg_loss, avg_iou, avg_precision, avg_recall


def main():
    args = parse_args()
    args = load_config(args)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Load transform and dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = RefCOCODataset(args.image_dir, args.annotation_file, args.refexp_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    print(f"📌 Loading model: {args.model_type} from {args.model_path}")
    model = get_model(args.model_type).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    loss_fn = torch.nn.SmoothL1Loss()

    # Run evaluation
    print("🔍 Evaluating on validation set...")
    val_loss, avg_iou, avg_precision, avg_recall = evaluate(
        model, dataloader, tokenizer, loss_fn, device
    )

    print("\n📊 External Evaluation Results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Average IoU:     {avg_iou:.4f}")
    print(f"Mean Precision:  {avg_precision:.4f}")
    print(f"Mean Recall:     {avg_recall:.4f}")


if __name__ == "__main__":
    main()
