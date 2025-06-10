import torch
import os
import numpy as np
import pandas as pd
from transformers import AutoFeatureExtractor, ASTForAudioClassification

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchaudio
from transformers import AutoModelForAudioClassification
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor
import pickle

import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modalities = ['rgb', 'spectrogram', 'eeg']
mod_path = r'C:\Users\user.DESKTOP-HI4HHBR\Downloads\facial_emotions_image_detection (1)'


class VideoAudioDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb = self.inputs['rgb'][idx]
        spec = self.inputs['spectrogram'][idx]
        eeg = self.inputs['eeg'][idx]
        label = self.labels[idx]
        return {'rgb': rgb, 'spectrogram': spec, 'eeg': eeg}, label


def prepare_dataloader(x, y, batch_size, shuffle=False):
    dataset = VideoAudioDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_layers, n_bottleneck_tokens, fusion_layer, model_eeg, model_aud, model_vid):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.fusion_layer = fusion_layer
        self.n_bottleneck_tokens = n_bottleneck_tokens
        self.encoders = nn.ModuleDict()

        self.dropout = nn.Dropout(p=0.1)  # Dropout for regularization

        for modality in modalities:
            if modality == 'rgb':
                self.encoders[modality] = model_vid.blocks
            elif modality == 'spectrogram':
                self.encoders[modality] = model_aud.blocks
            elif modality == 'eeg':
                self.encoders[modality] = model_eeg.transformer_layers
        self.norm_rgb = nn.LayerNorm(emb_dim)
        self.norm_spec = nn.LayerNorm(emb_dim)
        self.norm_eeg = nn.LayerNorm(emb_dim)

    def forward(self, inputs, bottleneck):
        x = inputs
        # assume rgb inputs of shape (batch x frames) x seq_len x emb_dim
        # audio inputs of shape batch x seq_len x emb_dim
        encoders = {}
        x_combined = []
        fusion_layer = self.fusion_layer

        for layer in range(self.num_layers):
            if layer < fusion_layer:
                for modality in modalities:
                    if modality == 'eeg':
                        layer_eeg = model_eeg.num_layers - (self.num_layers - layer)
                        if layer_eeg >= 0:
                            x[modality] = self.encoders[modality][layer_eeg](x[modality])
                    else:
                        x[modality] = self.encoders[modality][layer](x[modality])

            else:
                bottle = []
                for modality in modalities:
                    if modality == 'eeg':
                        #####################################################################################
                        layer_eeg = model_eeg.num_layers - (self.num_layers - layer)
                        if layer_eeg >= 0:
                            bottleneck_expanded = self.encoders[modality][layer_eeg].endsample(bottleneck)
                            bottleneck_expanded = nn.ReLU()(bottleneck_expanded)
                            t_mod = x[modality].shape[1]
                            in_mod = torch.cat([x[modality], bottleneck_expanded], dim=1)
                    else:
                        bottleneck_expanded = self.encoders[modality][layer].endsample(bottleneck)
                        bottleneck_expanded = nn.ReLU()(bottleneck_expanded)
                        if modality == 'rgb':
                            bottleneck_expanded = bottleneck_expanded.unsqueeze(1).expand(-1, 25, -1, -1).reshape(-1,
                                                                                                                  self.n_bottleneck_tokens,
                                                                                                                  self.emb_dim)
                        t_mod = x[modality].shape[1]
                        in_mod = torch.cat([x[modality], bottleneck_expanded], dim=1)

                    if modality == 'eeg':
                        layer_eeg = model_eeg.num_layers - (self.num_layers - layer)
                        if layer_eeg >= 0:
                            out_mod = self.encoders[modality][layer_eeg](in_mod)
                            x[modality] = out_mod[:, :t_mod]
                            bottleneck_eeg = out_mod[:, t_mod:]
                            #################################################################################
                            bottleneck_eeg = self.encoders[modality][layer_eeg].midsample(bottleneck_eeg)
                            bottleneck_eeg = nn.ReLU()(bottleneck_eeg)
                            bottle.append(bottleneck_eeg)
                    else:
                        out_mod = self.encoders[modality][layer](in_mod)
                        x[modality] = out_mod[:, :t_mod]
                        if modality == 'rgb':
                            bottleneck_reduced = torch.mean(
                                out_mod[:, t_mod:].view(-1, 25, self.n_bottleneck_tokens, self.emb_dim),
                                dim=1)  # average accross frames
                        else:
                            bottleneck_reduced = out_mod[:, t_mod:]
                        bottleneck_reduced = self.encoders[modality][layer].midsample(bottleneck_reduced)
                        bottleneck_reduced = nn.ReLU()(bottleneck_reduced)
                        bottle.append(bottleneck_reduced)

                bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)

        encoded_rgb = self.norm_rgb(x['rgb'])
        encoded_spec = self.norm_spec(x['spectrogram'])
        encoded = {
            'rgb': encoded_rgb,
            'spectrogram': encoded_spec,
            'eeg': x['eeg']
        }
        return encoded


class AMBT(nn.Module):
    def __init__(self, mlp_dim, num_classes, num_layers,
                 hidden_size, fusion_layer,model_eeg, model_aud, model_vid,
                 representation_size=None,
                 return_prelogits=False, return_preclassifier=False,
                 seperate=False
                 ):
        super(AMBT, self).__init__()

        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.representation_size = representation_size
        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier

        self.n_bottlenecks = 2
        self.bottleneck = nn.Parameter(torch.randn(1, self.n_bottlenecks * 3, 256) * 0.02)  # *3 because 3 modalities

        self.fusion_layer = fusion_layer

        self.model_eeg = model_eeg
        self.model_aud = model_aud
        self.model_vid = model_vid

        self.encoder = TransformerEncoder(
            self.hidden_size, self.num_layers,
            self.n_bottlenecks, self.fusion_layer, self.model_eeg,
            self.model_aud, self.model_vid
        )
        self.pos_embed_eeg = nn.Parameter(torch.zeros(1, 1 + 208, 768))
        self.pos_drop_video = nn.Dropout(p=0.0)
        self.pos_drop_audio = nn.Dropout(p=0.1)

        self.cls_token_aud = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed_aud = nn.Parameter(torch.zeros(1, 1 + 1212, 768))
        self.cls_token_vid = torch.load('cls_tokens.pth')
        self.pos_embed_vid = torch.load('position_embeddings.pth')
        self.temporal_encoder_audio = model_aud.patch_embed
        self.temporal_encoder_rgb = model_vid.patch_embed
        self.seperate = seperate

    def forward(self, x):

        # x should be a dict with rbg and spec as modalities (keys)
        # rgb input of shape (batch x channels x frames x image_size x image_size)
        # spectrogram input of shape (batch x channels x image_size x image_size)
        for modality in modalities:
            if modality == 'spectrogram':
                B = x[modality].shape[0]
                x[modality] = self.temporal_encoder_audio(x[modality])
                cls_tokens = self.cls_token_aud.expand(B, -1, -1)  # 복제
                x[modality] = torch.cat((cls_tokens, x[modality]), dim=1)  # 클래스 토큰 추가
                x[modality] = x[modality] + self.pos_embed_aud
                x[modality] = self.pos_drop_audio(x[modality])

            if modality == 'rgb':
                B = x[modality].shape[0]
                x[modality] = self.temporal_encoder_rgb(x[modality])
                cls_token_tensor = self.cls_token_vid
                pos_embed_tensor = self.pos_embed_vid

                # Ensure tensors are on the same device as the input tensor
                device = x[modality].device
                cls_token_tensor = cls_token_tensor.to(device)
                pos_embed_tensor = pos_embed_tensor.to(device)

                cls_tokens = cls_token_tensor.expand(B, -1, -1)
                x[modality] = torch.cat((cls_tokens, x[modality]), dim=1)  # Add class token
                x[modality] = x[modality] + pos_embed_tensor
                x[modality] = self.pos_drop_video(x[modality])
            if modality == 'eeg':
                x[modality] = self.model_eeg.conv1(x[modality])
                x[modality] = self.model_eeg.embedding(x[modality])

        batch, _, _ = x['spectrogram'].shape
        bottleneck_expanded = self.bottleneck.expand(batch, -1, -1)

        # now add cls token
        encoded = self.encoder(x, bottleneck=bottleneck_expanded)

        if self.return_preclassifier:
            return encoded

        x_out = {}
        counter = 0

        for modality in modalities:
            if modality == 'rgb':
                batch_size, _, _ = encoded['spectrogram'].size()
                cls_tok = encoded[modality][:, 0]
                cls_tok = self.model_vid.head(cls_tok)
                cls_tok = cls_tok.view(batch_size, 25, -1)  # (2,25,5)
                features = cls_tok.mean(dim=1)
                x_out[modality] = features
            if modality == 'spectrogram':
                x_out[modality] = encoded[modality][:, 0]
                x_out[modality] = self.model_aud.head(x_out[modality])
        x_pool = 0
        for modality in x_out:
            x_pool += x_out[modality]

        x_out['eeg'] = self.model_eeg.forward_ending(encoded['eeg'])
        x_pool += x_out['eeg']
        x_pool /= len(x_out)

        if self.seperate:
            return x_out

        if not self.training:
            return x_pool

        return x_out

