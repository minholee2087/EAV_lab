import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
from eegcnn_model import EEGCNN  # Import the pretrained model
import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from eegcnn_model import EEGCNN
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
    

# Classifier model
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=181, num_classes=5):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, S, feat_dim = x.shape  # not F !!
        x = x.view(B * S, feat_dim)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = x.view(B, S, -1)
        x = x.mean(dim=1)
        return x




# Training loop
def train_model(classifier, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50):
    

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
    
    

    print(te_x_eeg.shape, te_y_eeg.shape)
    te_x_eeg = torch.tensor(te_x_eeg) if not isinstance(te_x_eeg, torch.Tensor) else te_x_eeg
    te_y_eeg = torch.tensor(te_y_eeg) if not isinstance(te_y_eeg, torch.Tensor) else te_y_eeg

    test_dataset = TensorDataset(te_x_eeg, te_y_eeg)
    test_loader_2 = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model_2 = ShallowConvNet(nb_classes=5, Chans=30, Samples=500)
    model_path = os.path.join(r'D:\.spyder-py3\finetuned_cnn_7030(1)', f'eeg_finetuned_shallow_{sub}.pth')
    model_2.load_state_dict(torch.load(model_path))
    model_2=model_2.to(device)
    
    classifier.load_state_dict(torch.load(f"D:\\.spyder-py3\\Classifier_weights\\E_v\\classifier_vision_sub{sub}_new.pth"))

    classifier=classifier.to(device)
    
    
    model_2.eval()
    classifier.eval()
    correct_1 = 0  # Correct predictions for classifier
    correct_2 = 0  # Correct predictions for model
    correct_combined = 0  # Correct predictions for summed softmax outputs
    all_preds = []
    all_probs = []
    all_targets = []

    total_samples = 0  # Total number of samples

    with torch.no_grad():
        for (x_batch, y_batch), (x_batch_2, y_batch_2) in zip(test_loader, test_loader_2):
            # Ensure labels match
            assert torch.equal(y_batch, y_batch_2), "Mismatch in y_batch between test_loader and test_loader_2"
    
            total_samples += y_batch.size(0)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch_2, y_batch_2 = x_batch_2.to(device), y_batch_2.to(device)
    
            # Get model outputs
            outputs1 = classifier(x_batch)
            outputs2 = model_2(x_batch_2)
            
    
            # Compute softmax probabilities
            softmax2 = torch.nn.functional.softmax(outputs2, dim=1)
    
            # Get predictions
            predicted_1 = outputs1.argmax(dim=1)
            predicted_2 = softmax2.argmax(dim=1)
    
            # Accuracy per model
            correct_1 += (predicted_1 == y_batch).sum().item()
            correct_2 += (predicted_2 == y_batch_2).sum().item()
    
            # Combine softmax and get final predictions
            combined_outputs = torch.nn.functional.softmax(outputs1 + softmax2, dim=1)
            predicted_combined = combined_outputs.argmax(dim=1)
            correct_combined += (predicted_combined == y_batch).sum().item()
    
            # Collect outputs
            all_preds.append(predicted_combined.detach().cpu())
            all_probs.append(combined_outputs.detach().cpu())
            all_targets.append(y_batch.detach().cpu())
    
    # Stack predictions
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Compute overall metrics
    accuracy_1 = 100 * correct_1 / total_samples
    accuracy_2 = 100 * correct_2 / total_samples
    accuracy_combined = 100 * correct_combined / total_samples
    
    # F1 Score
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # AUC Score (multi-class)
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except ValueError:
        auc = -1
    
    # Print overall results
    print(f"Classifier Model Accuracy: {accuracy_1:.2f}%")
    print(f"Second Model Accuracy: {accuracy_2:.2f}%")
    print(f"Combined Softmax Accuracy: {accuracy_combined:.2f}%")
    print(f"Test Accuracy: {accuracy_combined / 100:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    
    # Emotion to valence/arousal mapping
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
    
    # Predicted valence/arousal
    pred_valence = np.array([emotion_to_valence[p] for p in all_preds])
    pred_arousal = np.array([emotion_to_arousal[p] for p in all_preds])
    
    # True valence/arousal
    true_valence = np.array([emotion_to_valence[t] for t in all_targets])
    true_arousal = np.array([emotion_to_arousal[t] for t in all_targets])
    
    # Valence metrics
    val_acc = accuracy_score(true_valence, pred_valence)
    val_f1 = f1_score(true_valence, pred_valence, average='binary')
    try:
        val_auc = roc_auc_score(true_valence, pred_valence)
    except ValueError:
        val_auc = -1
    
    # Arousal metrics
    aro_acc = accuracy_score(true_arousal, pred_arousal)
    aro_f1 = f1_score(true_arousal, pred_arousal, average='binary')
    try:
        aro_auc = roc_auc_score(true_arousal, pred_arousal)
    except ValueError:
        aro_auc = -1
    
    # Print valence/arousal results
    print(f"\nValence - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    print(f"Arousal - Acc: {aro_acc:.4f}, F1: {aro_f1:.4f}, AUC: {aro_auc:.4f}")
    with open('E_e_E_v(performance).txt', 'a') as f:
        f.write(f'Subject {sub} Valence Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}, Arousal Accuracy: {aro_acc:.4f}, F1-score: {aro_f1:.4f}, 5-class Accuracy: {accuracy_combined / 100:.4f}, F1: {f1:.4f}\n')


    # Save results
    
    


subs=range(1,43)

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

for sub in subs:    
    
    train_features, train_labels, test_features, test_labels = load_data(rf"D:\input images\Vision\subject_{sub:02d}_vis_unf.pkl")

    mean, std = train_features.mean(), train_features.std()
    test_features = (test_features - mean) / std
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    classifier = EmotionClassifier().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=5e-4)
    
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_epochs = 50
    
    train_model(classifier, train_loader, test_loader, criterion, optimizer, sub, num_epochs)
