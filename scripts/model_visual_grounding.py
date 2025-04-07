import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class VisualGroundingModel(nn.Module):
    def __init__(self, hidden_dim=512, num_decoder_layers=3):
        super(VisualGroundingModel, self).__init__()

        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.image_encoder = mobilenet.features
        self.image_proj = nn.Sequential(
            nn.Conv2d(960, hidden_dim, kernel_size=1),
            nn.ReLU()
        )

        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for layer in self.text_encoder.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, hidden_dim, 256))
        self.bbox_head = nn.Linear(hidden_dim, 4)

    def forward(self, image, text_input_ids, text_attention_mask):
        image_features = self.image_encoder(image)
        image_features = self.image_proj(image_features)
        B, C, H, W = image_features.shape
        image_features = image_features.flatten(2).permute(2, 0, 1)

        pos_enc = self.positional_encoding[:, :, :H*W].to(image_features.device)
        image_features = image_features + pos_enc.permute(2, 0, 1)

        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_outputs.last_hidden_state.permute(1, 0, 2)
        text_features = self.text_proj(text_features)

        decoder_output = self.decoder(text_features, image_features)
        attn_weights = torch.softmax(decoder_output, dim=0)
        weighted_output = (attn_weights * decoder_output).sum(dim=0)
        bbox = self.bbox_head(weighted_output)

        return bbox