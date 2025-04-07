import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from dataset import RefCOCODataset
from model_referring_grounding import ReferringGroundingModel
from utils import calculate_iou, calculate_map
from torchvision import transforms
from tqdm import tqdm

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset (val2014)
image_dir = 'data/val2014'
annotation_file = 'data/instances_val.json'
refexp_file = 'data/refs(unc)_val.p'
val_dataset = RefCOCODataset(image_dir, annotation_file, refexp_file, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ReferringGroundingModel().to(device)
model.load_state_dict(torch.load("models/visual_grounding_model.pth"))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bbox_loss_fn = torch.nn.SmoothL1Loss()

# Evaluation
val_loss = 0.0
all_pred_boxes = []
all_gt_boxes = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating on val2014"):
        images = batch['image'].to(device)
        bboxes = batch['bbox'].to(device)
        captions = batch['caption']
        encoding = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        predicted_bboxes, _ = model(images, input_ids, attention_mask)
        loss = bbox_loss_fn(predicted_bboxes, bboxes)
        val_loss += loss.item()

        all_pred_boxes.extend(predicted_bboxes.cpu())
        all_gt_boxes.extend(bboxes.cpu())

# Metrics
avg_val_loss = val_loss / len(val_loader)
precisions, recalls, ious = [], [], []

for pred_box, gt_box in zip(all_pred_boxes, all_gt_boxes):
    iou = calculate_iou(pred_box, gt_box)
    precision, recall = calculate_map([pred_box], [gt_box])
    precisions.append(precision)
    recalls.append(recall)
    ious.append(iou)

print(f"External Evaluation on val2014:")
print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Average IoU: {sum(ious) / len(ious):.4f}")
print(f"Mean Precision: {sum(precisions) / len(precisions):.4f}")
print(f"Mean Recall: {sum(recalls) / len(recalls):.4f}")