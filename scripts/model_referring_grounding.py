import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from transformers import DistilBertModel

class ReferringGroundingModel(nn.Module):
    def __init__(self, cnn_out_dim=256, text_out_dim=256):
        super().__init__()
        self.cnn = mobilenet_v2(pretrained=True).features
        self.conv_proj = nn.Conv2d(1280, cnn_out_dim, kernel_size=1)
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_proj = nn.Linear(768, text_out_dim)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(cnn_out_dim + text_out_dim, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.box_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + text_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, image, input_ids, attention_mask):
        B = image.size(0)
        x = self.cnn(image)
        x = self.conv_proj(x)

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_proj(text_outputs.last_hidden_state[:, 0, :])

        text_feat = text_feat.unsqueeze(-1).unsqueeze(-1)
        text_feat = text_feat.expand(-1, -1, x.size(2), x.size(3))

        fused = torch.cat([x, text_feat], dim=1)
        heatmap = self.heatmap_conv(fused).squeeze(1)

        prob = heatmap.view(B, -1)
        prob = F.softmax(prob, dim=-1)
        grid = self._make_grid(x.shape[2], x.shape[3], image.device)
        coords = torch.matmul(prob, grid)

        cx_norm = coords[:, 0] / (x.shape[3] - 1) * 2 - 1
        cy_norm = coords[:, 1] / (x.shape[2] - 1) * 2 - 1
        grid_coords = torch.stack([cx_norm, cy_norm], dim=1).unsqueeze(1).unsqueeze(1)
        sampled_feat = F.grid_sample(fused, grid_coords, align_corners=True).view(B, -1)
        wh = self.box_mlp(sampled_feat)

        cx, cy = coords[:, 0], coords[:, 1]
        scale = image.size(2) // x.size(2)
        cx = cx * scale
        cy = cy * scale
        w, h = wh[:, 0] * scale, wh[:, 1] * scale

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes, heatmap

    def _make_grid(self, H, W, device):
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        return grid