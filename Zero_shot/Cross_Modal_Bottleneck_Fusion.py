import torch
import torch.nn as nn
from Transformer_Audio import ViT_Encoder
from Transformer_Video import ViT_Encoder_Video

class CrossModalBottleneckFusion(nn.Module):
    def __init__(self, pretrained_audio_path, pretrained_video_path, n_bottlenecks=4, num_classes=5, output_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        self.n_bottlenecks = n_bottlenecks

        # Initialize encoders
        self.model_aud = ViT_Encoder(
            classifier=False,
            img_size=[1024, 128],
            in_chans=1,
            patch_size=(16, 16),
            stride=10,
            embed_pos=True
        )
        self.model_vid = ViT_Encoder_Video(
            classifier=False,
            img_size=(224, 224),
            in_chans=3,
            patch_size=(16, 16),
            stride=16,
            embed_pos=True
        )

        # Load only backbone weights
        def load_weights(model, path):
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            filtered_dict = {k: v for k, v in state_dict.items() if
                             not (k.startswith('head.') or k.startswith('classifier'))}
            model.load_state_dict(filtered_dict, strict=False)

        load_weights(self.model_aud, pretrained_audio_path)
        load_weights(self.model_vid, pretrained_video_path)

        # Move to device
        self.model_aud.to(device)
        self.model_vid.to(device)

        # Freeze encoders
        for p in self.model_aud.parameters():
            p.requires_grad = False
        for p in self.model_vid.parameters():
            p.requires_grad = False

        # Bottleneck tokens - one set for each modality
        self.audio_bottleneck = nn.Parameter(torch.randn(1, n_bottlenecks, 768))
        self.video_bottleneck = nn.Parameter(torch.randn(1, n_bottlenecks, 768))

        # Cross-attention layers for fusion
        self.audio_cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.video_cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Projection layers
        self.audio_proj = nn.Linear(768, 768)
        self.video_proj = nn.Linear(768, 768)

        # Classifier
        self.classifier = nn.Linear(768 * 2, num_classes)

        # Layer norms
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)

        # Add projection layer to match expected dimension
        self.projection = nn.Linear(768 * 2, output_dim)  # 1536 -> 256
        self.output_dim = output_dim

        # Move all parameters to device
        self.to(device)

    def forward(self, audio_input, video_input):
        B = audio_input.shape[0]

        # Extract audio features
        audio_feat = self.model_aud.feature(audio_input)  # (B, seq_len, 768)
        audio_cls = audio_feat[:, 0, :].unsqueeze(1)  # (B, 1, 768)

        # Extract video features
        B2, F, C, H, W = video_input.shape
        video_feat_list = []
        for b in range(B2):
            frame_out = self.model_vid.feature(video_input[b])  # (F, seq_len, 768)
            cls_tokens = frame_out[:, 0, :]  # CLS tokens for each frame
            avg_cls = cls_tokens.mean(dim=0)  # Average across frames
            video_feat_list.append(avg_cls)
        video_feat = torch.stack(video_feat_list)  # (B, 768)
        video_cls = video_feat.unsqueeze(1)  # (B, 1, 768)

        # Expand bottleneck tokens
        audio_bottleneck = self.audio_bottleneck.expand(B, -1, -1)  # (B, n_bottlenecks, 768)
        video_bottleneck = self.video_bottleneck.expand(B, -1, -1)  # (B, n_bottlenecks, 768)

        # Cross-modal attention: Audio attends to video bottleneck
        audio_fused, _ = self.audio_cross_attn(
            query=audio_bottleneck,
            key=video_bottleneck,
            value=video_bottleneck
        )
        audio_fused = self.norm1(audio_bottleneck + audio_fused)

        # Cross-modal attention: Video attends to audio bottleneck
        video_fused, _ = self.video_cross_attn(
            query=video_bottleneck,
            key=audio_bottleneck,
            value=audio_bottleneck
        )
        video_fused = self.norm2(video_bottleneck + video_fused)

        # Combine with original CLS tokens
        audio_rep = torch.cat([audio_cls, audio_fused.mean(dim=1, keepdim=True)], dim=1)
        video_rep = torch.cat([video_cls, video_fused.mean(dim=1, keepdim=True)], dim=1)

        # Final projections
        '''audio_rep = self.audio_proj(audio_rep.mean(dim=1))  # (B, 768)
        video_rep = self.video_proj(video_rep.mean(dim=1))  # (B, 768)

        # Concatenate for classification
        fused_feat = torch.cat([audio_rep, video_rep], dim=1)  # (B, 1536)
        logits = self.classifier(fused_feat)'''

        ###
        # Final projections
        audio_rep = self.audio_proj(audio_rep.mean(dim=1))  # (B, 768)
        video_rep = self.video_proj(video_rep.mean(dim=1))  # (B, 768)

        # Concatenate for classification
        fused_feat = torch.cat([audio_rep, video_rep], dim=1)  # (B, 1536)

        # Project to target dimension
        projected_feat = self.projection(fused_feat)  # (B, output_dim)

        logits = self.classifier(fused_feat)

        return logits, projected_feat, audio_rep, video_rep

        #return logits, audio_rep, video_rep