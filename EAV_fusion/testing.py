import pickle
import os
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
import sys
#change concat to mean in following file names, if want mean version
from model.AMBT_concat import AMBT
from unimodal_models.Transformer_Video_concat import ViT_Encoder_Video
from unimodal_models.Transformer_Audio_concat import ViT_Encoder_Audio, ast_feature_extract
from unimodal_models.Transformer_EEG_concat import EEG_Encoder

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






if __name__ == "__main__":
    for sub in range(1,43):
        model_aud = ViT_Encoder_Audio(classifier=True, img_size=[1024, 128], in_chans=1, patch_size=(16, 16), stride=10,
                                      embed_pos=True, fusion_layer=4)
        model_vid = ViT_Encoder_Video(classifier=True, img_size=(224, 224), in_chans=3, patch_size=(16, 16), stride=16,
                                      embed_pos=True, fusion_layer=4)
        num_layers = 4
        model_eeg = EEG_Encoder(nb_classes=5, Chans=30, Samples=500, num_layers=num_layers)


        fusion_layer=8    
        ambt = AMBT(
            mlp_dim=3072, num_classes=5, num_layers=12, 
            hidden_size=768, fusion_layer=fusion_layer, model_eeg=model_eeg,model_aud=model_aud,model_vid=model_vid,
            representation_size=256,
            return_prelogits=False, return_preclassifier=False
        )
        
        model_path = f'D:\.spyder-py3\AMBT_finetuned\dropout_sub{sub}_mbt_epoch10.pth'
        ambt.load_state_dict(torch.load(model_path), strict=False)
        
        
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
        
        
       
            
        
        
        
        criterion = torch.nn.CrossEntropyLoss()
        
        ambt = ambt.to(device)
        
        
        ambt.eval()
        correct, total = 0, 0
        outputs_batch = []
        true_labels_batch = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs['rgb'] = inputs['rgb'].view(-1, 3, 224, 224)
                inputs = {k: v.float().to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                outputs = ambt(inputs)
    
                # Collect predictions and true labels for F1-score calculation
                _, predicted = torch.max(outputs, dim=-1)
                outputs_batch.extend(predicted.cpu().numpy())  # Add predictions
                true_labels_batch.extend(labels.cpu().numpy())  # Add true labels
    
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
    
        test_accuracy = correct / total
        # Calculate F1-score
        f1 = f1_score(true_labels_batch, outputs_batch, average='weighted')
        
        print(f'Test accuracy of subject {sub} after epoch : {test_accuracy:.4f}')
        print(f'F1-score of subject {sub} after epoch : {f1:.4f}')
        with open("ambt_acc.txt", "a") as f:
            f.write(f'Test accuracy of subject {sub} : {test_accuracy:.4f}')
            f.write(f'F1-score of subject {sub} : {f1:.4f}')
            
                