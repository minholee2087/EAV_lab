import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch.nn.functional as F
from eegcnn_model import EEGCNN  
import numpy as np
import os

# Load dataset
def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def extract_features(model, eeg_data):
    print(eeg_data.shape)
    num_trials, num_segments, _, _ = eeg_data.shape
    eeg_data = eeg_data.view(-1, 7, 30)  # Reshape to (280*71, 7, 30) for batch processing
    with torch.no_grad():
        features = model(eeg_data)  # Output shape: (280*71, 181)
    features = features.view(num_trials, num_segments, 181)
    return features

# Optimized Model
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

# Optional: Xavier Initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Arousal/Valence Mapping
emotion_to_arousal = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0}
emotion_to_valence = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0}

def evaluate_extended(model, test_loader):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)  # [batch, num_classes]
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_probs.append(outputs.cpu())
            all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    test_acc = np.mean(all_preds == all_targets)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    try:
        test_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except ValueError:
        test_auc = -1.0

    pred_valence = np.array([emotion_to_valence[p] for p in all_preds])
    true_valence = np.array([emotion_to_valence[t] for t in all_targets])
    pred_arousal = np.array([emotion_to_arousal[p] for p in all_preds])
    true_arousal = np.array([emotion_to_arousal[t] for t in all_targets])

    val_acc = accuracy_score(true_valence, pred_valence)
    val_f1 = f1_score(true_valence, pred_valence)
    try:
        val_auc = roc_auc_score(true_valence, pred_valence)
    except ValueError:
        val_auc = -1.0

    aro_acc = accuracy_score(true_arousal, pred_arousal)
    aro_f1 = f1_score(true_arousal, pred_arousal)
    try:
        aro_auc = roc_auc_score(true_arousal, pred_arousal)
    except ValueError:
        aro_auc = -1.0

    return test_acc, test_f1, test_auc, val_acc, val_f1, val_auc, aro_acc, aro_f1, aro_auc

# Training one epoch
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total, correct, running_loss = 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)  # outputs [batch, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(loader), correct / total

# Run per subject
for subject in range(1, 43):
    sub=subject
    base_checkpoint_dir = r"C:\Users\user.DESKTOP-HI4HHBR\.spyder-py3\pycharm\ClassifaerEEG2Face\ClassifaerEEG2Face\checkpoints"
    model = EEGCNN(output_size=181)
        
    target_filename = f"best_model_{sub - 1}.pth"
    
    found = False
    for folder_name in os.listdir(base_checkpoint_dir):
        folder_path = os.path.join(base_checkpoint_dir, folder_name)
        if os.path.isdir(folder_path):
            potential_path = os.path.join(folder_path, target_filename)
            if os.path.exists(potential_path):
                print(f"Found checkpoint for subject {sub} in {folder_path}")
                state_dict = torch.load(potential_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                found = True
                break
    
    if not found:
        print(f"Checkpoint for subject {sub} not found in any subfolder.")
        
    file_name = f"subject_{sub:02d}_eeg_unfiltered_div.pkl"
    file_path = os.path.join(r"D:\input images\EEG", file_name)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            eeg_list = pickle.load(f)
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    # Unpack EEG data
    tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list
    
    # Convert to torch tensors
    tr_x_eeg = torch.tensor(tr_x_eeg[:,:71], dtype=torch.float32)  # (280, 71, 7, 30)
    te_x_eeg = torch.tensor(te_x_eeg[:,:71], dtype=torch.float32)  # (120, 71, 7, 30)
    
    
    # Function to extract features
    
    
    # Extract training & testing features
    train_features = extract_features(model, tr_x_eeg)  # Shape: (280*71, 181)
    test_features = extract_features(model, te_x_eeg)  # Shape: (120*71, 181)
    
    # Convert labels to tensor
    tr_y_eeg = torch.tensor(tr_y_eeg, dtype=torch.long)  # Shape: (280,)
    te_y_eeg = torch.tensor(te_y_eeg, dtype=torch.long)  # Shape: (120,)
    
    print("Extracted training features shape:", train_features.shape)  # (280, 71,181)
    print("Extracted testing features shape:", test_features.shape)    # (120, 71,181)
        
        
    batch_size = 8

    train_dataset = TensorDataset(train_features, tr_y_eeg)
    test_dataset = TensorDataset(test_features, te_y_eeg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EmotionClassifier().cuda()
    model.apply(init_weights)

    # Optional class balancing
    class_weights = torch.tensor(
        [len(tr_y_eeg) / (5 * (tr_y_eeg == i).sum().item()) for i in range(5)],
        dtype=torch.float
    ).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Uncomment if you want to train
    for epoch in range(100):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc, test_f1, test_auc, val_acc, val_f1, val_auc, aro_acc, aro_f1, aro_auc = evaluate_extended(model, test_loader)
        scheduler.step()
    
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"           Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        print(f"           Valence - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"           Arousal - Acc: {aro_acc:.4f}, F1: {aro_f1:.4f}, AUC: {aro_auc:.4f}")

    # Save if needed
    torch.save(model.state_dict(),  f"D:\.spyder-py3\Classifier_weights\E_r\classifier_model_sub{sub}_new.pth")
    with open('E_r(performance).txt', 'a') as f:
        f.write(f'Subject {subject} Valence Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}, Arousal Accuracy: {aro_acc:.4f}, F1-score: {aro_f1:.4f}, 5-class Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}\n')


