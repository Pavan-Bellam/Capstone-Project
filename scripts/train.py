import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from transformers import DistilBertTokenizer
from dataset import RefCOCODataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual/Referring Grounding Model")
    parser.add_argument('--config', type=str, help='Path to config JSON file')

    parser.add_argument('--model_type', type=str, default='referring', choices=['referring', 'visual'],
                        help="Choose the model: 'referring' or 'visual'")
    parser.add_argument('--run_num', type=int, default=1, help="Run number for this experiment")
    parser.add_argument('--image_dir', type=str, default='../data/train2014')
    parser.add_argument('--annotation_file', type=str, default='../data/instances.json')
    parser.add_argument('--refexp_file', type=str, default='../data/refs(unc).p')
    parser.add_argument('--output_model_path', type=str, default='../models/visual_grounding_model.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_split', type=float, default=0.93)
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


def get_dataloaders(args, transform):
    full_dataset = RefCOCODataset(args.image_dir, args.annotation_file, args.refexp_file, transform=transform)
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def train(model, dataloader, tokenizer, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        bboxes = batch['bbox'].to(device)
        captions = batch['caption']

        encoding = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        optimizer.zero_grad()
        predicted_bboxes, _ = model(images, input_ids, attention_mask)
        loss = loss_fn(predicted_bboxes, bboxes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, tokenizer, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            captions = batch['caption']

            encoding = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            predicted_bboxes, _ = model(images, input_ids, attention_mask)
            loss = loss_fn(predicted_bboxes, bboxes)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    args = load_config(args)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    os.makedirs(f"../results/", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"📦 Loading data from: {args.image_dir}")
    train_loader, val_loader = get_dataloaders(args, transform)

    print(f"📌 Initializing {args.model_type} model...")
    model = get_model(args.model_type).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{args.epochs}")

        train_loss = train(model, train_loader, tokenizer, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, tokenizer, loss_fn, device)

        print(f"📉 Training Loss: {train_loss:.4f}")
        print(f"📊 Validation Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Save model
    torch.save(model.state_dict(), args.output_model_path)
    print(f"\n✅ Model saved to {args.output_model_path}")

    # Save plot
    plot_path = f"results/loss_curve_run_{args.run_num}.png"
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Run {args.run_num} ({args.model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"📈 Loss curve saved to {plot_path}")

    # Save final results
    results_path = f"results/summary_run_{args.run_num}.json"
    with open(results_path, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "run_num": args.run_num,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate
        }, f, indent=4)
    print(f"📝 Summary saved to {results_path}")


if __name__ == "__main__":
    main()
