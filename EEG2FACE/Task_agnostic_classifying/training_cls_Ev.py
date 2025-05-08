import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch.nn.functional as F

import numpy as np

# Load dataset
def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

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
    train_features, train_labels, test_features, test_labels = load_data(rf"D:\input images\Vision\subject_{subject:02d}_vis_unf.pkl") # features after emoca model with dim (batch, num of windows, 181)

    # Normalize features
    mean, std = train_features.mean(), train_features.std()
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    train_features = train_features.float()
    test_features = test_features.float()
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=8, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=8, shuffle=False)

    model = EmotionClassifier().cuda()
    model.apply(init_weights)

    # Optional class balancing
    class_weights = torch.tensor(
        [len(train_labels) / (5 * (train_labels == i).sum().item()) for i in range(5)],
        dtype=torch.float
    ).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Uncomment if you want to train
    for epoch in range(50):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc, test_f1, test_auc, val_acc, val_f1, val_auc, aro_acc, aro_f1, aro_auc = evaluate_extended(model, test_loader)
        scheduler.step()
    
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"           Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        print(f"           Valence - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"           Arousal - Acc: {aro_acc:.4f}, F1: {aro_f1:.4f}, AUC: {aro_auc:.4f}")

    # Save if needed
    torch.save(model.state_dict(), f"D:\\.spyder-py3\\Classifier_weights\\E_v\\classifier_vision_sub{subject}_new.pth")
    with open('E_v(performance).txt', 'a') as f:
        f.write(f'Subject {subject} Valence Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}, Arousal Accuracy: {aro_acc:.4f}, F1-score: {aro_f1:.4f}, 5-class Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}\n')


