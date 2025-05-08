from Dataload_eeg import *
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import os
import pickle 
from torch.utils.data import DataLoader, TensorDataset

# Define the EEGNet model
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import torch.optim as optim


import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 40, (1, 13))
        self.conv2 = nn.Conv2d(40, 40, (Chans, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1)
        
        # Pooling and dropout
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(dropoutRate)
        
        # Fully connected layer
        self.fc = nn.Linear(40 * 1 * 65, nb_classes)
        
        # Constraints
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        
        # Apply square activation
        x = torch.square(x)
        
        x = self.pool(x)
        
        # Apply log activation
        x = torch.log(torch.clamp(x, min=1e-7, max=10000))
        
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return F.softmax(x, dim=1)



import numpy as np


if __name__ == "__main__":
    from sklearn.metrics import f1_score, roc_auc_score

    # Emotion → Valence & Arousal mappings
    emotion_to_arousal = {
        0: 0,  # Neutral -> Low Arousal
        1: 0,  # Sadness -> Low Arousal
        2: 1,  # Anger -> High Arousal
        3: 1,  # Happiness -> High Arousal
        4: 0,  # Calmness -> Low Arousal
    }
    emotion_to_valence = {
        0: 0,  # Neutral -> Positive Valence
        1: 1,  # Sadness -> Negative Valence
        2: 1,  # Anger -> Negative Valence
        3: 0,  # Happiness -> Positive Valence
        4: 0,  # Calmness -> Positive Valence
    }
    result_acc = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(1,43):
        file_name = f"subject_{i:02d}_eeg.pkl"
        file_ = os.path.join(r"D:\input images\EEG", file_name)
        if os.path.exists(file_):
            with open(file_, 'rb') as f:
                eeg_list = pickle.load(f)
        else:
            print('Does not exist')
        
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list
        
        #to torch and reshape        
        tr_x_eeg = torch.from_numpy(tr_x_eeg).float().unsqueeze(1).to(device)  # Reshape to (batch, 1, chans, samples)
        tr_y_eeg = torch.tensor(tr_y_eeg, dtype=torch.long).to(device)
        te_x_eeg = torch.from_numpy(te_x_eeg).float().unsqueeze(1).to(device)  # Reshape to (batch, 1, chans, samples)   
        te_y_eeg = torch.tensor(te_y_eeg, dtype=torch.long).to(device)
        print(tr_x_eeg.shape)
        # Create DataLoader
        train_dataset = TensorDataset(tr_x_eeg, tr_y_eeg)
        test_dataset = TensorDataset(te_x_eeg, te_y_eeg)
        
        # Parameters
        num_epochs = 480
        norm_rate = 1.0
        batch_size = 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        #model = EEGNet_tor(nb_classes=5, Chans=30, Samples=500)


        model = ShallowConvNet(nb_classes=5, Chans=30, Samples=500)
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            model.train()
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
                with torch.no_grad():
                    model.fc.weight.data = torch.renorm(model.fc.weight.data, p=2, dim=0, maxnorm=0.5)
        
            # Evaluation
            model.eval()
            all_preds = []
            all_targets = []
            all_probs = []
        
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    outputs = model(batch_data)
                    probs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(probs, 1)
        
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        
            # Primary classification metrics
            acc = (np.array(all_preds) == np.array(all_targets)).mean()
            f1 = f1_score(all_targets, all_preds, average='macro')
            try:
                auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
            except ValueError:
                auc = float('nan')
        
            # Valence & Arousal conversion
            arousal_preds = [emotion_to_arousal[p] for p in all_preds]
            arousal_targets = [emotion_to_arousal[t] for t in all_targets]
            valence_preds = [emotion_to_valence[p] for p in all_preds]
            valence_targets = [emotion_to_valence[t] for t in all_targets]
        
            # Arousal metrics (binary)
            arousal_acc = (np.array(arousal_preds) == np.array(arousal_targets)).mean()
            arousal_f1 = f1_score(arousal_targets, arousal_preds, average='binary')
            try:
                arousal_auc = roc_auc_score(arousal_targets, arousal_preds)
            except ValueError:
                arousal_auc = float('nan')
        
            # Valence metrics (binary)
            valence_acc = (np.array(valence_preds) == np.array(valence_targets)).mean()
            valence_f1 = f1_score(valence_targets, valence_preds, average='binary')
            try:
                valence_auc = roc_auc_score(valence_targets, valence_preds)
            except ValueError:
                valence_auc = float('nan')
        
            # Print all results
            print(f"[Epoch {epoch}] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            print(f"              Arousal Acc: {arousal_acc:.4f} | F1: {arousal_f1:.4f} | AUC: {arousal_auc:.4f}")
            print(f"              Valence Acc: {valence_acc:.4f} | F1: {valence_f1:.4f} | AUC: {valence_auc:.4f}")
        
            result_acc.append(acc)
        
        with open('E_e(performance).txt', 'a') as f:
            f.write(f'Subject {i} Valence Accuracy: {valence_acc:.4f}, F1-score: {valence_f1:.4f}, Arousal Accuracy: {arousal_acc:.4f}, F1-score: {arousal_f1:.4f}, 5-class Accuracy: {acc:.4f}, F1: {f1:.4f}\n')
            
        path=os.path.join(r'D:\.spyder-py3\finetuned_cnn_7030(1)', f'eeg_finetuned_shallow_{i}_new.pth')
        torch.save(model.state_dict(), path)
        
