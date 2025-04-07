import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer
from dataset import RefCOCODataset
from model_referring_grounding import ReferringGroundingModel
from torchvision import transforms
from tqdm import tqdm
import os

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset (train2014)
image_dir = '../data/train2014'
annotation_file = '../data/instances.json'
refexp_file = '../data/refs(unc).p'
full_dataset = RefCOCODataset(image_dir, annotation_file, refexp_file, transform=transform)

# Internal validation split
train_size = int(0.93 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Model, tokenizer, loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ReferringGroundingModel().to(device)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bbox_loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters())

# Train
train_epoch_loss = []
val_epoch_loss = []

for epoch in range(50):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/50"):
        images = batch['image'].to(device)
        bboxes = batch['bbox'].to(device)
        captions = batch['caption']

        encoding = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        optimizer.zero_grad()
        predicted_bboxes, _ = model(images, input_ids, attention_mask)
        loss = bbox_loss_fn(predicted_bboxes, bboxes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_epoch_loss.append(avg_epoch_loss)
    print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss:.4f}")

    # Evaluate on internal val set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            captions = batch['caption']
            encoding = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            predicted_bboxes, _ = model(images, input_ids, attention_mask)
            loss = bbox_loss_fn(predicted_bboxes, bboxes)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_epoch_loss.append(avg_val_loss)
    print(f"Epoch {epoch + 1} Internal Validation Loss: {avg_val_loss:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "../models/visual_grounding_model.pth")
print("Model saved to ../models/visual_grounding_model.pth")