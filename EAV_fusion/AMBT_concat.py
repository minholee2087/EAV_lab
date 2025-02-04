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
from Transformer_Video_concat import ViT_Encoder_Video
from Transformer_Audio_concat import ViT_Encoder_Audio, ast_feature_extract
from Transformer_EEG_concat import ShallowConvNet
import pickle

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score


            
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



mod_path = r'C:\Users\user.DESKTOP-HI4HHBR\Downloads\facial_emotions_image_detection (1)'

for sub in range(1,43):
        
    fusion_layer=8    

    
    model_aud = ViT_Encoder_Audio(classifier=True, img_size=[1024, 128], in_chans=1, patch_size=(16, 16), stride=10, embed_pos=True,fusion_layer=fusion_layer)
    model_path = f"D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_audio_finetuned_{sub}.pth"
    model_aud.load_state_dict(torch.load(model_path), strict=False)

    model_vid = ViT_Encoder_Video(classifier=True, img_size=(224, 224), in_chans=3, patch_size=(16, 16), stride=16, embed_pos=True,fusion_layer=fusion_layer)
    model_path = f"D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_video_finetuned_{sub}.pth"
    model_vid.load_state_dict(torch.load(model_path), strict=False)
    
    num_layers=4
    model_eeg = ShallowConvNet(nb_classes=5, Chans=30, Samples=500, num_layers=num_layers)
    path= os.path.join(r'EEG_finetuned_models', f'subject_{sub:02d}_layers4_epochs400.pth')
    model_eeg.load_state_dict(torch.load(path), strict=False)
    
    file_name = f"subject_{sub:02d}_vis.pkl"
    file_ = os.path.join(r"D:\input images\Vision", file_name)

    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
    tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2
    data_video = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
    processor = AutoImageProcessor.from_pretrained(mod_path)
    
    pixel_values_list = []
    for img_set in tr_x_vis:
        for img in img_set:
            processed = processor(images=img, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze()
            pixel_values_list.append(pixel_values)
    
    vals = torch.stack(pixel_values_list)
    
    pixel_values_list = []
    for img_set in te_x_vis:
        for img in img_set:
            processed = processor(images=img, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze()
            pixel_values_list.append(pixel_values)
    
    vals_test = torch.stack(pixel_values_list)
    
    direct=r"D:\input images\Audio"
    file_name = f"subject_{sub:02d}_aud.pkl"
    file_ = os.path.join(direct, file_name)

    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
    tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2
    tr_x_aud_ft = ast_feature_extract(tr_x_vis)
    te_x_aud_ft = ast_feature_extract(te_x_vis)
    tr_y_aud=tr_y_vis
    te_y_aud=te_y_vis
    
    file_name = f"subject_{sub:02d}_eeg.pkl"
    file_ = os.path.join(r"D:\input images\EEG", file_name)
    if os.path.exists(file_):
        with open(file_, 'rb') as f:
            eeg_list = pickle.load(f)
    else:
        print('Does not exist')
        
    tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list      
    tr_x_eeg = torch.from_numpy(tr_x_eeg).float().unsqueeze(1) # Reshape to (batch, 1, chans, samples)
    te_x_eeg = torch.from_numpy(te_x_eeg).float().unsqueeze(1) # Reshape to (batch, 1, chans, samples)   
    data_eeg= [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]
    
    data = [
        {'rgb': vals.view(-1, 25, 3, 224, 224), 
        'spectrogram': torch.tensor(tr_x_aud_ft.unsqueeze(1), dtype=torch.float32),
        'eeg':tr_x_eeg},
        torch.from_numpy(tr_y_aud).long(), 
        {'rgb': vals_test.view(-1, 25, 3, 224, 224), 
        'spectrogram': torch.tensor(te_x_aud_ft.unsqueeze(1), dtype=torch.float32),
        'eeg':te_x_eeg},
        torch.from_numpy(te_y_aud).long(), 
    ]
    
    tr_x, tr_y, te_x, te_y = data
    
    train_dataloader = prepare_dataloader(tr_x, tr_y, batch_size=2, shuffle=True)
    test_dataloader = prepare_dataloader(te_x, te_y, batch_size=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = ['rgb', 'spectrogram', 'eeg']
    
    
    class TransformerEncoder(nn.Module):
        def __init__(self, emb_dim, num_layers, n_bottleneck_tokens, fusion_layer):
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
                            #x[modality] = self.dropout(x[modality])  # Apply dropout
                            
                else: 
                    bottle = []
                    batch, _, latent = bottleneck.shape
                    bottleneck = bottleneck.view(batch, -1, 3 * latent)
    
                    for modality in modalities:  
                        if modality == 'eeg':
                            layer_eeg = model_eeg.num_layers - (self.num_layers - layer)
                            if layer_eeg >= 0:
                                
                                bottleneck_expanded = self.encoders[modality][layer_eeg].endsample(bottleneck)
                                bottleneck_expanded = (nn.ReLU()(bottleneck_expanded))  # Dropout and activation
                                t_mod = x[modality].shape[1]
                                in_mod = torch.cat([x[modality], bottleneck_expanded], dim=1)
                        else:
                            bottleneck_expanded = self.encoders[modality][layer].endsample(bottleneck)
                            bottleneck_expanded = (nn.ReLU()(bottleneck_expanded))  # Dropout and activation
                            if modality == 'rgb':
                                bottleneck_expanded = bottleneck_expanded.unsqueeze(1).expand(-1, 25, -1, -1).reshape(-1, self.n_bottleneck_tokens, self.emb_dim)
                            t_mod = x[modality].shape[1]
                            in_mod = torch.cat([x[modality], bottleneck_expanded], dim=1)
    
                        if modality == 'eeg':
                            layer_eeg = model_eeg.num_layers - (self.num_layers - layer)
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

    
    class MBT(nn.Module):
        def __init__(self, mlp_dim, num_classes, num_layers, 
                     hidden_size, fusion_layer, 
                     representation_size=None, 
                     return_prelogits=False, return_preclassifier=False,
                     ):
            super(MBT, self).__init__()
    
            self.mlp_dim = mlp_dim
            self.num_classes = num_classes
            self.hidden_size = hidden_size
            self.num_layers = num_layers 
            self.representation_size = representation_size
            self.return_prelogits = return_prelogits
            self.return_preclassifier = return_preclassifier
    
    
            #AutoModelForImageClassification.from_pretrained(mod_path).vit.embeddings
    
            self.n_bottlenecks = 2
            self.bottleneck = nn.Parameter(torch.randn(1, self.n_bottlenecks*3, 256) * 0.02) # *3 because 3 modalities
    
            self.fusion_layer = fusion_layer 
            
            self.encoder = TransformerEncoder(
                self.hidden_size, self.num_layers,  
                self.n_bottlenecks, self.fusion_layer 
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
            self.model_eeg=model_eeg
            self.model_aud=model_aud
            self.model_vid=model_vid
    
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
                    cls_tok=self.model_vid.head(cls_tok)
                    cls_tok = cls_tok.view(batch_size, 25, -1) #(2,25,5)
                    features = cls_tok.mean(dim=1)
                    x_out[modality] = features
                if modality == 'spectrogram':
                    x_out[modality] = encoded[modality][:, 0]
                    x_out[modality] = self.model_aud.head(x_out[modality])
            x_pool = 0 
            for modality in x_out: 
                #x_out[modality] = self.output_projection[modality](x_out[modality]) 
                x_pool += x_out[modality]
                
            #print(f"x shape before permute: {encoded['eeg'].shape}")
            
            x_out['eeg'] = self.model_eeg.forward_ending(encoded['eeg'])
            x_pool += x_out['eeg']
            x_pool /= len(x_out)
            if not self.training: 
                return x_pool 
            return x_out 
        
    mbt = MBT(
        mlp_dim=3072, num_classes=5, num_layers=12, 
        hidden_size=768, fusion_layer=fusion_layer, representation_size=256,
        return_prelogits=False, return_preclassifier=False
    )
    
    
    
    criterion = torch.nn.CrossEntropyLoss()
    
    mbt = mbt.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        mbt = nn.DataParallel(mbt)
    
    
    modes=[True,False]
    
    for freeze in modes:
        
        

        
        for param in mbt.module.parameters():
            param.requires_grad = True
            
        for param in mbt.module.encoder.parameters():
            param.requires_grad = False
        for param in mbt.module.temporal_encoder_audio.parameters():
            param.requires_grad = False
        for param in mbt.module.temporal_encoder_rgb.parameters():
            param.requires_grad = False
        for param in mbt.module.model_eeg.parameters():
            param.requires_grad = False
            
        mbt.module.cls_token_aud.requires_grad = False
        mbt.module.cls_token_vid.requires_grad = False
        mbt.module.pos_embed_aud.requires_grad = False
        mbt.module.pos_embed_vid.requires_grad = False
        mbt.module.bottleneck.requires_grad=False
        
        
            
        if freeze:
            epochs = 5
            
            optimizer = optim.Adam(mbt.parameters(), lr=5e-5)
            
            
        else:
            
            epochs = 20
            
            modalities1 = ['rgb', 'spectrogram']
            for modality in modalities1: 
                for layer_idx in range(8, 12):  # Layers 8 to 12
                    layer = mbt.module.encoder.encoders[modality][layer_idx]
                    for param in layer.midsample.parameters():
                        param.requires_grad = True
                    for param in layer.endsample.parameters():
                        param.requires_grad = True
                        
            for layer_idx in range(0, 4):  # Layers 8 to 12
                layer = mbt.module.encoder.encoders['eeg'][layer_idx]
                for param in layer.midsample.parameters():
                    param.requires_grad = True
                for param in layer.endsample.parameters():
                    param.requires_grad = True
            optimizer = optim.Adam(mbt.parameters(), lr=5e-5)
            
        
        for epoch in range(1, epochs+1): 
            mbt.train()
            train_correct, train_total = 0, 0
            running_loss = 0.0 
            
            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                inputs['rgb'] = inputs['rgb'].view(-1, 3, 224, 224)
                
                inputs = {k: v.float().to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = mbt(inputs)
                if isinstance(outputs, dict):
                    ce_loss = []
                    for mod in outputs:
                        ce_loss.append(criterion(outputs[mod], labels))
                else:
                    break         
                    
                loss = torch.mean(torch.stack(ce_loss))
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                if i % 1 == 0: 
                    #print(f'Epoch {epoch}, loss: {running_loss / 10:.4f}')
                    running_loss = 0 
        
            mbt.eval()
            correct, total = 0, 0
            outputs_batch = []
            true_labels_batch = []
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs['rgb'] = inputs['rgb'].view(-1, 3, 224, 224)
                    inputs = {k: v.float().to(device) for k, v in inputs.items()}
                    labels = labels.to(device)
                    outputs = mbt(inputs)
        
                    # Collect predictions and true labels for F1-score calculation
                    _, predicted = torch.max(outputs, dim=-1)
                    outputs_batch.extend(predicted.cpu().numpy())  # Add predictions
                    true_labels_batch.extend(labels.cpu().numpy())  # Add true labels
        
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
        
            test_accuracy = correct / total
            # Calculate F1-score
            f1 = f1_score(true_labels_batch, outputs_batch, average='weighted')
            
            print(f'Test accuracy after epoch {epoch}: {test_accuracy:.4f}')
            print(f'F1-score after epoch {epoch}: {f1:.4f}')
            
            with open('bottleneck_from_scratch_old+eeg_transf40_sep_concatbottle_0.txt', 'a') as f:
                f.write(f'Subject {sub} Epoch {epoch} freeze:{freeze} Testing Accuracy: {test_accuracy:.4f} F1-score: {f1:.4f} \n')

            if epoch >= 10 and epoch <= 18:
                torch.save(mbt.module.state_dict(), f'D:\.spyder-py3\AMBT_finetuned\dropout_sub{sub}_mbt_epoch{epoch}.pth')

        