import torch 
import numpy as np 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

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





def mlp(dim, hidden_dim, output_dim, layers=1, activation="relu"):
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    seq = [nn.Linear(dim, hidden_dim), act_fn()]
    for _ in range(layers - 1):
        seq += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
    seq.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*seq)


class InfoNCECritic(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers=1, activation="relu", temperature=0.1):
        super().__init__()
        self.scorer = mlp(in_dim, hidden_dim, 1, layers, activation)
        self.temperature = temperature

    def forward(self, x, y):
        n = y.size(0)
        x_tile = x.unsqueeze(0).expand(n, -1, -1)
        y_tile = y.unsqueeze(1).expand(-1, n, -1)

        pos = self.scorer(torch.cat([x, y], dim=-1)) / self.temperature
        all_pairs = self.scorer(torch.cat([x_tile, y_tile], dim=-1)) / self.temperature

        lower_bound = pos.mean() - (all_pairs.logsumexp(dim=1).mean() - np.log(n))
        return -lower_bound


class CLUBCritic(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers=1, activation="relu", temperature=0.1):
        super().__init__()
        self.scorer = mlp(in_dim, hidden_dim, 1, layers, activation)
        self.temperature = temperature

    def forward(self, x, y):
        n = y.size(0)
        x_tile = x.unsqueeze(0).expand(n, -1, -1)
        y_tile = y.unsqueeze(1).expand(-1, n, -1)

        pos = self.scorer(torch.cat([x, y], dim=-1)) / self.temperature
        neg = self.scorer(torch.cat([x_tile, y_tile], dim=-1)) / self.temperature

        return pos.mean() - neg.mean()



class FACL(nn.Module):
    def __init__(self, x1_dim, x2_dim, y_dim, hidden_dim=128, embed_dim=64,
                 layers=1, activation="relu", lr=1e-4, alpha=1e-4, temperature=0.1):
        super().__init__()
        self.y_dim = y_dim
        self.lr = lr
        self.alpha = alpha

        # Encoders
        self.enc_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation)
        self.enc_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation)

        # Critics
        self.infonce = InfoNCECritic(embed_dim * 2, hidden_dim, layers, activation, temperature)
        self.club = CLUBCritic(embed_dim * 2 + y_dim, hidden_dim, layers, activation, temperature)

    def ohe(self, y):
        N = y.size(0)
        y_ohe = torch.zeros(N, self.y_dim, device=y.device)
        y_ohe[torch.arange(N), y.view(-1).long()] = 1
        return y_ohe

    def forward(self, x1, x2, y):
        # Normalize embeddings
        x1_embed = F.normalize(self.enc_x1(x1), dim=-1)
        x2_embed = F.normalize(self.enc_x2(x2), dim=-1)
        y_ohe = self.ohe(y)

        loss_uncond = self.infonce(x1_embed, x2_embed)
        loss_cond = self.club(torch.cat([x1_embed, y_ohe], dim=-1), x2_embed)

        # Downweight contrastive loss
        return self.alpha * (loss_uncond + loss_cond)





class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_layers, n_bottleneck_tokens, fusion_layer,model_eeg,model_aud,model_vid):
        super().__init__()
        
        self.emb_dim = emb_dim 
        self.num_layers = num_layers
        self.fusion_layer = fusion_layer 
        self.n_bottleneck_tokens = n_bottleneck_tokens
        self.encoders = nn.ModuleDict()
        self.eeg_layers=model_eeg.num_layers
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
            fusion_layer = self.fusion_layer  
            bottleneck_expanded={}
            for layer in range(self.num_layers):
                if layer < fusion_layer: 
                    for modality in modalities: 
                        if modality == 'eeg':
                            layer_eeg = self.eeg_layers - (self.num_layers - layer)
                            if layer_eeg >= 0:
                                x[modality] = self.encoders[modality][layer_eeg](x[modality])
                        else:
                            x[modality] = self.encoders[modality][layer](x[modality])
                            #x[modality] = self.dropout(x[modality])  # Apply dropout
                            
                else: 
                    bottle = []
                    batch, _, latent = bottleneck.shape
                    bottleneck = bottleneck.view(batch, -1, 3 * latent)
    
                    for modality in modalities:  
                        if modality == 'eeg':
                            layer_eeg = self.eeg_layers - (self.num_layers - layer)
                            if layer_eeg >= 0:
                                
                                bottleneck = self.encoders[modality][layer_eeg].endsample(bottleneck)
                                bottleneck = (nn.ReLU()(bottleneck))  # Dropout and activation
                                if layer>fusion_layer:
                                    bottleneck_expanded[modality] = bottleneck_expanded[modality]+bottleneck
                                else:
                                    bottleneck_expanded[modality] = bottleneck
                                t_mod = x[modality].shape[1]
                                in_mod = torch.cat([x[modality], bottleneck_expanded[modality]], dim=1)
                        else:
                            bottleneck = self.encoders[modality][layer].endsample(bottleneck)
                            bottleneck = (nn.ReLU()(bottleneck))  # Dropout and activation
                            if modality == 'rgb':
                                bottleneck_exp = bottleneck.unsqueeze(1).expand(-1, 25, -1, -1).reshape(-1, self.n_bottleneck_tokens, self.emb_dim)
                                if layer>fusion_layer:
                                    bottleneck_expanded[modality] = bottleneck_expanded[modality]+bottleneck_exp
                                else:
                                    bottleneck_expanded[modality] = bottleneck_exp
                            else:
                                if layer>fusion_layer:
                                    bottleneck_expanded[modality] = bottleneck_expanded[modality]+bottleneck
                                else:
                                    bottleneck_expanded[modality] = bottleneck
                            t_mod = x[modality].shape[1]
                            in_mod = torch.cat([x[modality], bottleneck_expanded[modality]], dim=1)
    
                        if modality == 'eeg':
                            layer_eeg = self.eeg_layers - (self.num_layers - layer)
                            if layer_eeg >= 0:
                                out_mod = self.encoders[modality][layer_eeg](in_mod)
                                x[modality] = out_mod[:, :t_mod]
                                bottleneck_eeg = out_mod[:, t_mod:]
                                bottleneck_eeg = (nn.ReLU()(self.encoders[modality][layer_eeg].midsample(bottleneck_eeg)))
                                bottle.append(bottleneck_eeg)
                        else:
                            out_mod = self.encoders[modality][layer](in_mod)
                            x[modality] = out_mod[:, :t_mod]
                            if modality == 'rgb':
                                bottleneck_reduced = torch.mean(out_mod[:, t_mod:].view(-1, 25, self.n_bottleneck_tokens, self.emb_dim), dim=1)
                            else: 
                                bottleneck_reduced = out_mod[:, t_mod:]
                            bottleneck_reduced = (nn.ReLU()(self.encoders[modality][layer].midsample(bottleneck_reduced)))
                            bottle.append(bottleneck_reduced)
    
                    bottleneck = torch.cat(bottle, dim=1)  # Concatenate along sequence dimension
                  
            encoded_rgb = self.norm_rgb(x['rgb'])
            encoded_spec = self.norm_spec(x['spectrogram'])
            encoded = {
                'rgb': encoded_rgb,
                'spectrogram': encoded_spec,
                'eeg': x['eeg']
            }
            return encoded


class AMBT_FACL(nn.Module):
    def __init__(self, mlp_dim, num_classes, num_layers, 
                 hidden_size, fusion_layer, model_eeg,model_aud,model_vid):
        super(AMBT_FACL, self).__init__()

        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers 


        self.n_bottlenecks = 2
        self.bottleneck = nn.Parameter(torch.randn(1, self.n_bottlenecks*3, 256) * 0.02) # *3 because 3 modalities

        self.fusion_layer = fusion_layer 
        
        self.model_eeg=model_eeg
        self.model_aud=model_aud
        self.model_vid=model_vid
        
        self.encoder = TransformerEncoder(
            self.hidden_size, self.num_layers,  
            self.n_bottlenecks, self.fusion_layer , self.model_eeg,
            self.model_aud, self.model_vid
        )
        self.pos_embed_eeg = nn.Parameter(torch.zeros(1, 1 + 208, 768))
        self.pos_drop_video = nn.Dropout(p=0.0)
        self.pos_drop_audio = nn.Dropout(p=0.1)
        
        self.cls_token_aud = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed_aud = nn.Parameter(torch.zeros(1, 1 + 1212, 768))
        self.cls_token_vid = torch.load('D:\.spyder-py3\cls_tokens.pth')
        self.pos_embed_vid = torch.load('D:\.spyder-py3\position_embeddings.pth')
        self.temporal_encoder_audio = model_aud.patch_embed
        self.temporal_encoder_rgb = model_vid.patch_embed
        
        self.lossav= FACL(x1_dim=768, x2_dim=768, y_dim=5, hidden_dim=256, embed_dim=768)
        self.lossev= FACL(x1_dim=2600, x2_dim=768, y_dim=5, hidden_dim=256, embed_dim=768)
        self.lossae= FACL(x1_dim=768, x2_dim=2600, y_dim=5, hidden_dim=256, embed_dim=768)


    def forward(self, x, y):
        

        
        
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

        x_out = {}

        for modality in modalities: 
            if modality == 'rgb':
                batch_size, _, _ = encoded['spectrogram'].size()
                cls_tok = encoded[modality][:, 0]
                vid_feat=cls_tok.view(batch_size, 25, -1)
                vid_feat = vid_feat.mean(dim=1)
                cls_tok=self.model_vid.head(cls_tok)
                cls_tok = cls_tok.view(batch_size, 25, -1) #(2,25,5)
                features = cls_tok.mean(dim=1)
                x_out[modality] = features
            if modality == 'spectrogram':
                x_out[modality] = encoded[modality][:, 0]
                aud_feat = x_out[modality]
                x_out[modality] = self.model_aud.head(x_out[modality])

        eeg_feat=self.model_eeg.feature_ending(encoded['eeg'])
        
        L11= self.lossav.forward(aud_feat,vid_feat,y)
        L21= self.lossev.forward(eeg_feat,vid_feat,y)
        L31= self.lossae.forward(aud_feat,eeg_feat,y)
        
        loss_contr=(0.7*(L11)+0.2*(L21)+0.1*(L31))
        
        x_pool = 0 
        for modality in x_out: 
            x_pool += x_out[modality]
            
        
        x_out['eeg'] = self.model_eeg.forward_ending(encoded['eeg'])
        x_pool += x_out['eeg']
        x_pool /= len(x_out)
        
        
        if not self.training: 
            return x_pool 
        
        return x_out, loss_contr
    
    
