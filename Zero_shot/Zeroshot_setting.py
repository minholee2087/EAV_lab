from Transformer_Video import ViT_Encoder_Video
from Transformer_Audio import ViT_Encoder, ast_feature_extract
import pickle
from torchvision import transforms
import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as FF
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class FusionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super(FusionNN, self).__init__()
        # 첫 번째 히든 레이어
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)

        # 두 번째 히든 레이어
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)

        # 출력 레이어
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)

        x = self.fc3(x)  # 최종 출력 (로짓 형태)
        return x

class ZeroShotModel(nn.Module):
    def __init__(self, eeg_dim, shared_dim=256, num_classes=5):
        super(ZeroShotModel, self).__init__()

        # EEG pathway
        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_dim, 512),
            nn.ReLU(),
            nn.Linear(512, shared_dim),
        )

        # Classification layer for EEG shared space
        self.eeg_cls = nn.Linear(shared_dim, num_classes)

    def forward_eeg(self, eeg_input):
        # Map EEG input to shared space and classify
        eeg_shared = self.eeg_proj(eeg_input)
        eeg_cls_output = self.eeg_cls(eeg_shared)
        return eeg_shared, eeg_cls_output

def load_subject_data(directory, subject_idx, audio=True, vision=True, eeg=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_images(image_list):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images
            transforms.ToTensor(),  # Convert to Tensor and scale [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        pixel_values_list = []
        for img_set in image_list:  # Loop through batches
            for img in img_set:  # Loop through frames
                pil_image = Image.fromarray(img.astype('uint8'))
                processed_img = preprocess(pil_image)  # Apply transformations
                pixel_values_list.append(processed_img)

        # Stack and reshape to (batch_size, num_frames, channels, height, width)
        batch_size = len(image_list)
        num_frames = len(image_list[0])
        channels, height, width = 3, 224, 224  # Fixed dimensions
        return torch.stack(pixel_values_list).view(batch_size, num_frames, channels, height, width)

    # Initialize all data to empty tensors/lists
    tr_x_aud_ft = torch.tensor([]).to(device)
    tr_y_aud = torch.tensor([], dtype=torch.long).to(device)
    te_x_aud_ft = torch.tensor([]).to(device)
    te_y_aud = torch.tensor([], dtype=torch.long).to(device)

    processed_tr_x_vis = torch.tensor([]).to(device)
    tr_y_vis = torch.tensor([], dtype=torch.long).to(device)
    processed_te_x_vis = torch.tensor([]).to(device)
    te_y_vis = torch.tensor([], dtype=torch.long).to(device)

    tr_x_eeg = torch.tensor([]).to(device)
    tr_y_eeg = torch.tensor([], dtype=torch.long).to(device)
    te_x_eeg = torch.tensor([]).to(device)
    te_y_eeg = torch.tensor([], dtype=torch.long).to(device)

    if audio:
        # Audio data
        audio_path = os.path.join(directory, "Audio", f"subject_{subject_idx:02d}_aud.pkl")
        with open(audio_path, 'rb') as f:
            aud_list = pickle.load(f)

        tr_x_aud, tr_y_aud_, te_x_aud, te_y_aud_ = aud_list
        tr_y_aud = torch.tensor(tr_y_aud_, dtype=torch.long).to(device)
        te_y_aud = torch.tensor(te_y_aud_, dtype=torch.long).to(device)

        # Process audio features
        tr_x_aud_ft = ast_feature_extract(tr_x_aud).unsqueeze(1).to(device)
        te_x_aud_ft = ast_feature_extract(te_x_aud).unsqueeze(1).to(device)

    if vision:
        # Vision2 data
        vision_path = os.path.join(directory, "Vision", f"subject_{subject_idx:02d}_vis.pkl")
        with open(vision_path, 'rb') as f:
            vis_list = pickle.load(f)

        tr_x_vis, tr_y_vis_, te_x_vis, te_y_vis_ = vis_list
        tr_y_vis = torch.tensor(tr_y_vis_, dtype=torch.long).to(device)
        te_y_vis = torch.tensor(te_y_vis_, dtype=torch.long).to(device)

        # Process vision data
        processed_tr_x_vis = preprocess_images(tr_x_vis).to(device)
        processed_te_x_vis = preprocess_images(te_x_vis).to(device)

    if eeg:
        # EEG data
        eeg_path = os.path.join(directory, "EEG", f"subject_{subject_idx:02d}_eeg.pkl")
        with open(eeg_path, 'rb') as f:
            eeg_list = pickle.load(f)

        tr_x_eeg_, tr_y_eeg_, te_x_eeg_, te_y_eeg_ = eeg_list
        tr_y_eeg = torch.tensor(tr_y_eeg_, dtype=torch.long).to(device)
        te_y_eeg = torch.tensor(te_y_eeg_, dtype=torch.long).to(device)

        # Convert EEG data to PyTorch tensors
        tr_x_eeg = torch.from_numpy(tr_x_eeg_).float().unsqueeze(1).to(device)  # Reshape to (batch, 1, chans, samples)
        te_x_eeg = torch.from_numpy(te_x_eeg_).float().unsqueeze(1).to(device)  # Reshape to (batch, 1, chans, samples)

    # Return all preprocessed data
    return (
        tr_x_aud_ft, tr_y_aud,
        te_x_aud_ft, te_y_aud,
        processed_tr_x_vis, tr_y_vis,
        processed_te_x_vis, te_y_vis,
        tr_x_eeg, tr_y_eeg,
        te_x_eeg, te_y_eeg
    )

def prepare_multilabel_data(Data_eeg):
    """
    Generate high/low arousal and positive/negative valence labels based on emotion classes.

    Args:
        tr_x_eeg (Tensor): Training EEG data.
        tr_y_eeg (Tensor): Training EEG emotion labels.
        te_x_eeg (Tensor): Test EEG data.
        te_y_eeg (Tensor): Test EEG emotion labels.

    Returns:
        dict: A dictionary containing EEG features and their corresponding labels:
            - 'tr_x_eeg': Training EEG features
            - 'tr_y_emotion': Training emotion labels
            - 'tr_y_arousal': Training arousal labels
            - 'tr_y_valence': Training valence labels
            - 'te_x_eeg': Test EEG features
            - 'te_y_emotion': Test emotion labels
            - 'te_y_arousal': Test arousal labels
            - 'te_y_valence': Test valence labels
    """

    tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data_eeg
    # Emotion label mappings
    emotion_to_arousal = {
        0: 0,  # Neutral -> Low Arousal
        1: 0,  # Sadness -> Low Arousal
        2: 1,  # Anger -> High Arousal
        3: 1,  # Happiness -> High Arousal
        4: 0,  # Calmness -> Low Arousal
    }

    emotion_to_valence = {
        0: 0,  # Neutral -> Negative
        1: 0,  # Sadness -> Negative
        2: 0,  # Anger -> Negative
        3: 1,  # Happiness -> Positive
        4: 1,  # Calmness -> Positive
    }

    # Generate arousal and valence labels for training data
    tr_y_arousal = torch.tensor([emotion_to_arousal[label.item()] for label in tr_y_eeg], dtype=torch.long)
    tr_y_valence = torch.tensor([emotion_to_valence[label.item()] for label in tr_y_eeg], dtype=torch.long)

    # Generate arousal and valence labels for test data
    te_y_arousal = torch.tensor([emotion_to_arousal[label.item()] for label in te_y_eeg], dtype=torch.long)
    te_y_valence = torch.tensor([emotion_to_valence[label.item()] for label in te_y_eeg], dtype=torch.long)

    # Return dictionary of data
    return tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence, te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence

def load_models(base_dir, subject_idx):
    model_aud = ViT_Encoder(
        classifier=True,
        img_size=[1024, 128],
        in_chans=1,
        patch_size=(16, 16),
        stride=10,
        embed_pos=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio_model_path = os.path.join(base_dir, "Audio", f"model_with_weights_audio_finetuned_{subject_idx}.pth")
    state_dict_aud = torch.load(audio_model_path, map_location=torch.device(device), weights_only=True)
    model_aud.load_state_dict(state_dict_aud)
    model_aud = model_aud.to(device)
    model_aud.eval()

    # Vision2 모델 복원
    model_vis = ViT_Encoder_Video(
        classifier=True,
        img_size=(224, 224),
        in_chans=3,
        patch_size=(16, 16),
        stride=16,
        embed_pos=True
    )
    vision_model_path = os.path.join(base_dir, "Vision2", f"model_with_weights_video_finetuned_{subject_idx}.pth")
    state_dict_vis = torch.load(vision_model_path, map_location=torch.device(device), weights_only=True)
    model_vis.load_state_dict(state_dict_vis)
    #state_dict_vis['cls_token']

    model_vis = model_vis.to(device)
    model_vis.eval()

    # Fusion 모델 복원
    model_av = FusionNN(input_dim=2 * 768, hidden_dim1=256, hidden_dim2=32, output_dim=5).to(device)
    fusion_model_path = os.path.join(base_dir, "AudioVision", f"subject_{subject_idx:02d}_av_finetune_model.pth")
    state_dict_av = torch.load(fusion_model_path, map_location=torch.device(device), weights_only=True)
    model_av.load_state_dict(state_dict_av)
    model_av = model_av.to(device)
    model_av.eval()

    return model_aud, model_vis, model_av

def predict_eeg(te_x_av, te_y_av, model_eeg, batch_size=32):
    model_eeg.eval()

    # Prepare test data loader
    test_dataset = torch.utils.data.TensorDataset(te_x_av, te_y_av)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize counters
    correct = 0
    total = 0

    # Evaluation loop
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Move data to the same device as the model
            batch_x, batch_y = batch_x.to(next(model_eeg.parameters()).device), batch_y.to(next(model_eeg.parameters()).device)
            # Forward pass
            outputs = model_eeg(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            # Update total and correct counts
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    # Calculate accuracy
    accuracy = correct / total
    print(accuracy)
    return accuracy

def predict_av(te_x_vis, te_x_aud_ft, model_vis, model_aud, fusion_model, label, tsne = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move models to the appropriate device
    model_vis = model_vis.to(device)
    model_aud = model_aud.to(device)
    fusion_model = fusion_model.to(device)

    # Ensure all models are in evaluation mode
    model_vis.eval()
    model_aud.eval()
    fusion_model.eval()

    with torch.no_grad():
        # Extract features from vision data
        B, F, C, H, W = te_x_vis.shape
        vis_features = []
        for b in range(B):  # Process each batch separately
            frame_outputs = model_vis.feature(te_x_vis[b, :, :, :, :])  # Shape: (F, num_tokens, feature_dim)
            frame_class_tokens = frame_outputs[:, 0, :]  # Extract class token for all frames
            averaged_features = frame_class_tokens.mean(dim=0)  # Average across frames
            vis_features.append(averaged_features)
        vis_features = torch.stack(vis_features)  # Shape: (B, feature_dim)

        # Extract features from audio data
        aud_features = model_aud.feature(te_x_aud_ft)[:, 0, :]  # Shape: (B, feature_dim)

        # Concatenate features
        combined_features = torch.cat((vis_features, aud_features), dim=1)  # Shape: (B, combined_feature_dim)

        # Extract intermediate features (fc1) from the fusion model
        fusion_intermediate = fusion_model.fc1(combined_features)  # Shape: (B, intermediate_dim)

        # Pass through the rest of the model for final prediction
        predictions = fusion_model(combined_features)  # Shape: (B, num_classes)

        # Get predicted classes
        predicted_classes = torch.argmax(predictions, dim=1)  # Shape: (B,)

        # Compute accuracy
        accuracy = (predicted_classes == label).float().mean().item()

    # Visualize using t-SNE
    if tsne:
        sne = TSNE(n_components=2, random_state=42)
        tsne_features = tsne.fit_transform(fusion_intermediate.to(device).numpy())

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=label.to(device).numpy(), cmap='viridis', alpha=0.8)
        plt.colorbar(scatter, label='Class Labels')
        plt.title("t-SNE Visualization of Fusion Model's Intermediate Space (fc1)")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.show()

    return accuracy

def contrastive(Data, Models, optimizer=None, epochs=15, batch_size=32, temperature=0.07, alpha=0.5, beta=0.5, gamma=0.5, freeze_epochs=2):

    def contrastive_loss(features_1, features_2, temperature=0.07):
        # Normalize features
        features_1 = FF.normalize(features_1, p=2, dim=1)
        features_2 = FF.normalize(features_2, p=2, dim=1)

        # Compute cosine similarity
        logits = torch.mm(features_1, features_2.t()) / temperature
        labels = torch.arange(features_1.size(0)).to(features_1.device)

        # Cross entropy loss
        loss = FF.cross_entropy(logits, labels)
        return loss

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Unpack data and models
    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data
    model_aud, model_vis, model_eeg, fusion_model = Models

    # Move models to device
    model_vis = model_vis.to(device)
    model_aud = model_aud.to(device)
    model_eeg = model_eeg.to(device)
    fusion_model = fusion_model.to(device)

    # Freeze audio, vision, and fusion models
    for param in model_vis.parameters():
        param.requires_grad = False
    for param in model_aud.parameters():
        param.requires_grad = False
    for param in fusion_model.parameters():
        param.requires_grad = False

    # Add shared layers to both models
    shared_layer = nn.Linear(256, 256).to(device)  # Common shared space
    eeg_projection_layer = nn.Linear(2600, 256).to(device)  # Map EEG features to 256 dimensions
    shared_to_class = nn.Linear(256, 5).to(device)  # Classification layer for shared space (5 classes)

    # Activation function
    activation_fn = nn.ReLU()

    # Optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(
            list(shared_layer.parameters()) +
            list(eeg_projection_layer.parameters()) +
            list(shared_to_class.parameters()), lr=0.001
        )

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(tr_x_aud), torch.tensor(tr_x_vis), torch.tensor(tr_x_eeg), torch.tensor(tr_y_eeg)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor(te_x_aud), torch.tensor(te_x_vis), torch.tensor(te_x_eeg), torch.tensor(te_y_eeg)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        model_vis.eval()
        model_aud.eval()
        fusion_model.eval()

        # Freeze/unfreeze EEG model based on epoch
        if epoch < freeze_epochs:
            for param in model_eeg.parameters():
                param.requires_grad = False
        else:
            for param in model_eeg.parameters():
                param.requires_grad = True
            # Add EEG parameters to optimizer after unfreezing
            if epoch == freeze_epochs:
                optimizer.add_param_group({"params": model_eeg.parameters()})

        model_eeg.train()

        total_loss_epoch = 0
        for batch in train_loader:
            batch_x_aud, batch_x_vis, batch_x_eeg, batch_y = batch
            batch_x_aud, batch_x_vis, batch_x_eeg, batch_y = (
                batch_x_aud.to(device),
                batch_x_vis.to(device),
                batch_x_eeg.to(device),
                batch_y.to(device),
            )

            # Reset gradients
            optimizer.zero_grad()

            # Extract AV features
            B, _, C, H, W = batch_x_vis.shape
            vis_features = []
            for b in range(B):
                frame_outputs = model_vis.feature(batch_x_vis[b])  # Shape: (F, num_tokens, feature_dim)
                frame_class_tokens = frame_outputs[:, 0, :]
                averaged_features = frame_class_tokens.mean(dim=0)  # Average across frames
                vis_features.append(averaged_features)
            vis_features = torch.stack(vis_features)  # Shape: (B, feature_dim)
            aud_features = model_aud.feature(batch_x_aud)[:, 0, :]
            combined_features_av = torch.cat((vis_features, aud_features), dim=1)

            # Pass AV features through fusion model and shared layer
            fusion_intermediate = fusion_model.fc1(combined_features_av)
            fusion_shared_output = activation_fn(shared_layer(fusion_intermediate))

            # Pass EEG features through projection and shared layer
            eeg_projected = activation_fn(eeg_projection_layer(model_eeg.feature(batch_x_eeg)))
            eeg_shared_output = activation_fn(shared_layer(eeg_projected))

            # Classification in shared space
            shared_class_output = shared_to_class(eeg_shared_output)
            loss_classification_shared = FF.cross_entropy(shared_class_output, batch_y)

            # Contrastive Loss
            loss_contrastive = contrastive_loss(fusion_shared_output, eeg_shared_output)

            # EEG-specific classification loss
            eeg_classification_output = model_eeg(batch_x_eeg)
            loss_classification_eeg = FF.cross_entropy(eeg_classification_output, batch_y)

            # Total loss
            total_loss = loss_classification_shared + loss_contrastive + alpha * loss_classification_eeg

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss_epoch:.4f}")

        # Evaluation loop
        model_eeg.eval()
        shared_correct = 0
        eeg_correct = 0
        total = 0

        with torch.no_grad():
            for batch_x_aud, batch_x_vis, batch_x_eeg, batch_y in test_loader:
                batch_x_eeg, batch_y = batch_x_eeg.to(device), batch_y.to(device)

                # Shared space accuracy
                shared_outputs = shared_to_class(
                    activation_fn(shared_layer(eeg_projection_layer(model_eeg.feature(batch_x_eeg))))
                )
                _, shared_predicted = torch.max(shared_outputs.data, 1)
                shared_correct += (shared_predicted == batch_y).sum().item()

                # EEG model's original softmax accuracy
                eeg_outputs = model_eeg(batch_x_eeg)
                _, eeg_predicted = torch.max(eeg_outputs.data, 1)
                eeg_correct += (eeg_predicted == batch_y).sum().item()

                total += batch_y.size(0)

        shared_accuracy = shared_correct / total
        eeg_accuracy = eeg_correct / total

        print(f"Shared Space Test Accuracy: {100 * shared_accuracy:.2f}%")
        print(f"EEG Model Test Accuracy: {100 * eeg_accuracy:.2f}%")

    # Return updated EEG model
    return model_eeg

def zeroshot_training(Data, Models, ZeroShotModel, optimizer=None, epochs=15, lr=0.005, batch_size=32, eeg_model_freeze = "unfreeze", fusion_model_freeze = "freeze"):

    def info_nce_loss(features_1, features_2, temperature=0.07):
        # Normalize features
        features_1 = FF.normalize(features_1, p=2, dim=1)
        features_2 = FF.normalize(features_2, p=2, dim=1)

        # Concatenate features
        batch_size = features_1.size(0)
        combined_features = torch.cat([features_1, features_2], dim=0)  # Shape: (2 * batch_size, feature_dim)

        # Compute cosine similarity
        logits = torch.mm(combined_features, combined_features.t()) / temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=features_1.device).bool()
        logits.masked_fill_(mask, float('-inf'))

        # Create labels
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(
            features_1.device)

        # Cross entropy loss
        loss = FF.cross_entropy(logits, labels)
        return loss

    def centroid_loss(features, labels, centroids):
        """
        Compute the distance between features and their respective class centroids.
        """
        total_loss = 0
        for label in torch.unique(labels):
            class_mask = labels == label
            class_features = features[class_mask]
            centroid = centroids[label.item()]
            total_loss += torch.sum((class_features - centroid) ** 2) / len(class_features)
        return total_loss

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Unpack data and models
    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data
    model_aud, model_vis, model_eeg, fusion_model = Models

    # Move models to device
    model_vis = model_vis.to(device)
    model_aud = model_aud.to(device)
    model_eeg = model_eeg.to(device)
    fusion_model = fusion_model.to(device)
    zero_shot_model = ZeroShotModel.to(device)

    # Freeze audio, vision, and fusion models
    for param in model_vis.parameters():
        param.requires_grad = False
    for param in model_aud.parameters():
        param.requires_grad = False
    for param in fusion_model.parameters():
        param.requires_grad = False
    for param in model_eeg.parameters():
        param.requires_grad = False

    if fusion_model_freeze == 'unfreeze':
        for param in fusion_model.parameters():
            param.requires_grad = True

    if eeg_model_freeze == 'unfreeze':
        for param in model_eeg.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam([
            {'params': zero_shot_model.parameters()},
            {'params': model_eeg.parameters(), 'lr': 0.001}  # Lower learning rate for fine-tuning
        ], lr=0.005)

    # Optimizer for ZeroShotModel
    if optimizer is None:
        optimizer = torch.optim.Adam(zero_shot_model.parameters(), lr= lr)

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.clone(tr_x_aud).detach(), torch.clone(tr_x_vis).detach(), torch.clone(tr_x_eeg).detach(), torch.clone(tr_y_eeg).detach()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        torch.clone(te_x_aud).detach(), torch.clone(te_x_vis).detach(), torch.clone(te_x_eeg).detach(), torch.clone(te_y_eeg).detach()
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_sums = {}
    class_counts = {}
    # Training loop
    for epoch in range(epochs):
        # Only train ZeroShotModel
        zero_shot_model.train()

        total_loss_epoch = 0
        for batch in train_loader:
            batch_x_aud, batch_x_vis, batch_x_eeg, batch_y = batch
            batch_x_aud, batch_x_vis, batch_x_eeg, batch_y = (
                batch_x_aud.to(device),
                batch_x_vis.to(device),
                batch_x_eeg.to(device),
                batch_y.to(device),
            )
            # Reset gradients
            optimizer.zero_grad()
            # Extract AV features
            B, _, C, H, W = batch_x_vis.shape
            vis_features = []
            for b in range(B):
                frame_outputs = model_vis.feature(batch_x_vis[b])  # Shape: (F, num_tokens, feature_dim)
                frame_class_tokens = frame_outputs[:, 0, :]
                averaged_features = frame_class_tokens.mean(dim=0)  # Average across frames
                vis_features.append(averaged_features)
            vis_features = torch.stack(vis_features)  # Shape: (B, feature_dim)

            # Extract audio features
            aud_features = model_aud.feature(batch_x_aud)[:, 0, :]  # Shape: (B, feature_dim)
            combined_features_av = torch.cat((vis_features, aud_features), dim=1)  # Concatenate AV features
            # Pass AV features through fusion model to get shared space
            av_shared = fusion_model.fc1(combined_features_av)  # Intermediate AV shared representation (fixed)

            # Update centroids dynamically
            # Accumulative centroid updates
            unique_labels = batch_y.unique()
            for label in unique_labels:
                label_mask = batch_y == label
                class_features = av_shared[label_mask]

                if label.item() not in class_sums:
                    class_sums[label.item()] = torch.zeros_like(class_features[0])
                    class_counts[label.item()] = 0

                # Update running sums and counts
                class_sums[label.item()] += class_features.sum(dim=0)
                class_counts[label.item()] += class_features.size(0)

            # Compute current centroids
            class_centroids = {
                label: class_sums[label] / class_counts[label]
                for label in class_sums.keys()
            }

            # Extract EEG features
            eeg_features = model_eeg.feature(batch_x_eeg)
            eeg_shared, eeg_class = zero_shot_model.forward_eeg(eeg_features)
            av_logits = fusion_model.fc2(av_shared)  # Output logits for classification (shape: B, num_classes)

            # Compute losses
            loss_classification_eeg = FF.cross_entropy(eeg_class, batch_y)
            loss_classification_av = FF.cross_entropy(av_logits, batch_y)
            loss_contrastive = info_nce_loss(eeg_shared, av_shared, temperature=0.07)
            loss_centroid = centroid_loss(eeg_shared, batch_y, class_centroids)

            # Total loss
            total_loss = loss_contrastive + loss_classification_eeg + loss_classification_av + loss_centroid

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()

        # Evaluation loop
        zero_shot_model.eval()
        model_eeg.eval()
        # Initialize lists to store predictions for each method
        all_av_predictions = []
        all_eeg_orig_predictions = []
        all_eeg_centroid_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x_aud, batch_x_vis, batch_x_eeg, batch_y in test_loader:
                # Move data to device
                batch_x_aud, batch_x_vis, batch_x_eeg, batch_y = (
                    batch_x_aud.to(device),
                    batch_x_vis.to(device),
                    batch_x_eeg.to(device),
                    batch_y.to(device),
                )

                # Extract vision features
                B, F, C, H, W = batch_x_vis.shape
                vis_features = []
                for b in range(B):  # Process each batch separately
                    frame_outputs = model_vis.feature(batch_x_vis[b])  # Shape: (F, num_tokens, feature_dim)
                    frame_class_tokens = frame_outputs[:, 0, :]  # Extract class token for all frames
                    averaged_features = frame_class_tokens.mean(dim=0)  # Average across frames
                    vis_features.append(averaged_features)
                vis_features = torch.stack(vis_features)  # Shape: (B, feature_dim)

                # Extract audio features
                aud_features = model_aud.feature(batch_x_aud)[:, 0, :]  # Shape: (B, feature_dim)
                combined_features = torch.cat((vis_features, aud_features), dim=1)  # Shape: (B, combined_feature_dim)

                # Get shared AV features (fixed)
                av_shared = fusion_model(combined_features)

                # Extract EEG features and pass through ZeroShotModel
                eeg_features = model_eeg.feature(batch_x_eeg)
                eeg_shared, eeg_class = zero_shot_model.forward_eeg(eeg_features)

                # Classification using shared space
                _, av_predicted = torch.max(av_shared, dim=1)
                _, eeg_orig_predicted = torch.max(model_eeg(batch_x_eeg)[0], dim=1)

                # Centroid-based classification for EEG
                eeg_centroid_predicted = []
                for i, eeg_feature in enumerate(eeg_shared):
                    distances = []
                    for label, centroid in class_centroids.items():
                        # Compute Euclidean distance
                        distance = torch.norm(eeg_feature - centroid)
                        distances.append((distance.item(), label))
                    # Assign the class with the minimum distance
                    predicted_label = min(distances, key=lambda x: x[0])[1]
                    eeg_centroid_predicted.append(predicted_label)

                # Store predictions and labels
                all_av_predictions.extend(av_predicted.to(device).tolist())
                all_eeg_orig_predictions.extend(eeg_orig_predicted.to(device).tolist())
                all_eeg_centroid_predictions.extend(eeg_centroid_predicted)
                all_labels.extend(batch_y.to(device).tolist())

        # Convert to tensors for consistent handling
        all_av_predictions = torch.tensor(all_av_predictions)
        all_eeg_orig_predictions = torch.tensor(all_eeg_orig_predictions)
        all_eeg_centroid_predictions = torch.tensor(all_eeg_centroid_predictions)
        all_labels = torch.tensor(all_labels)
        # Calculate and print accuracies in one statement
        print(
            f"Audio-Vision Accuracy: {(all_av_predictions == all_labels).sum().item() / 120 * 100:.2f}%, "
            f"EEG Accuracy: {(all_eeg_orig_predictions == all_labels).sum().item() / 120 * 100:.2f}%, "
            f"EEG Centroid-Based Accuracy: {(all_eeg_centroid_predictions == all_labels).sum().item() / 120 * 100:.2f}%"
        )
        # Return predictions and labels
    return all_av_predictions, all_eeg_orig_predictions, all_eeg_centroid_predictions, all_labels

def predict_zeroshot_e_av(Data, Models, model_zs):

    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, \
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, \
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data

    model_aud, model_vis, model_eeg, fusion_model = Models

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move models to device
    model_vis = model_vis.to(device)
    model_aud = model_aud.to(device)
    model_eeg = model_eeg.to(device)
    fusion_model = fusion_model.to(device)
    model_zs = model_zs.to(device)

    # Ensure models are in evaluation mode
    model_vis.eval()
    model_aud.eval()
    model_eeg.eval()
    fusion_model.eval()
    model_zs.eval()

    # DataLoader for evaluation
    test_dataset = TensorDataset(tr_x_aud, tr_x_vis, tr_x_eeg, tr_y_eeg)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize metrics
    total = 0
    correct_eeg = 0
    correct_av_centroid = 0

    av_shared_features = []
    av_shared_labels = []
    with torch.no_grad():
        for batch_x_aud, batch_x_vis, batch_x_eeg, batch_y in test_loader:
            # Move data to device
            batch_x_aud, batch_x_vis, batch_x_eeg, batch_y = (
                batch_x_aud.to(device),
                batch_x_vis.to(device),
                batch_x_eeg.to(device),
                batch_y.to(device),
            )

            # Extract vision features
            B, F, C, H, W = batch_x_vis.shape
            vis_features = []
            for b in range(B):  # Process each batch separately
                frame_outputs = model_vis.feature(batch_x_vis[b])  # Shape: (F, num_tokens, feature_dim)
                frame_class_tokens = frame_outputs[:, 0, :]  # Extract class token for all frames
                averaged_features = frame_class_tokens.mean(dim=0)  # Average across frames
                vis_features.append(averaged_features)
            vis_features = torch.stack(vis_features)  # Shape: (B, feature_dim)

            # Extract audio features
            aud_features = model_aud.feature(batch_x_aud)[:, 0, :]  # Shape: (B, feature_dim)

            # Combine vision and audio features
            combined_features = torch.cat((vis_features, aud_features), dim=1)  # Shape: (B, combined_feature_dim)

            # Get shared AV features (fixed centroid)
            av_shared = fusion_model.fc1(combined_features)  # Shape: (B, shared_dim)

            av_shared_features.append(av_shared.to(device))
            av_shared_labels.append(batch_y.to(device))
        av_shared_features = torch.cat(av_shared_features)  # Shape: (total_samples, shared_dim)
        av_shared_labels = torch.cat(av_shared_labels)  # Shape: (total_samples,)

        # Calculate class-wise centroids
        unique_labels = av_shared_labels.unique()
        class_centroids = {}

        for label in unique_labels:
            label_mask = av_shared_labels == label  # Boolean mask for current label
            class_features = av_shared_features[label_mask]  # Select features for this class
            centroid = class_features.mean(dim=0)  # Compute mean across all samples in the class
            class_centroids[label.item()] = centroid  # Store centroid for this class

        # Use AV centroids to classify EEG data
        te_x_eeg, te_y_eeg = te_x_eeg.to(device), te_y_eeg.to(device)

        with torch.no_grad():
            # Extract EEG features
            eeg_features = model_eeg.feature(te_x_eeg)  # Shape: (total_samples, eeg_feature_dim)

            # Map EEG features to shared AV space
            eeg_shared, _ = model_zs.forward_eeg(eeg_features)  # Shape: (total_samples, shared_dim)

            # Classify using AV centroids
            eeg_preds = []
            # Loop over each EEG trial
            for feature in eeg_shared:  # feature: (shared_dim,)
                # Calculate distances to all AV centroids
                distances = []
                for label, centroid in class_centroids.items():
                    centroid = centroid.to(feature.device)
                    distance = torch.norm(feature - centroid)  # Compute L2 distance
                    distances.append((label, distance))  # Store label and distance

                # Find the label of the nearest centroid
                nearest_label = min(distances, key=lambda x: x[1])[0]  # Get label with smallest distance
                eeg_preds.append(nearest_label)

            # Convert predictions to tensor
            eeg_preds = torch.tensor(eeg_preds, device=device)

            # Calculate training accuracy
            correct = (eeg_preds == te_y_eeg).sum().item()
            total = te_y_eeg.size(0)
            training_accuracy = correct / total
            print(training_accuracy)
    return eeg_preds

def prepare_zeroshot_data(Data, exclude_class = 0 ):
    """
    Excludes the first class from training data but keeps all classes in test data.
    """
    tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, \
    tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, \
    tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = Data

    # Exclude the first class (class 0) from training data
    train_mask = tr_y_eeg != exclude_class  # Boolean mask to filter training data
    tr_x_aud = tr_x_aud[train_mask]
    tr_y_aud = tr_y_aud[train_mask]
    tr_x_vis = tr_x_vis[train_mask]
    tr_y_vis = tr_y_vis[train_mask]
    tr_x_av = tr_x_eeg[train_mask]
    tr_y_av = tr_y_eeg[train_mask]

    # Test data remains unchanged
    return (tr_x_aud, tr_y_aud, te_x_aud, te_y_aud,
            tr_x_vis, tr_y_vis, te_x_vis, te_y_vis,
            tr_x_av, tr_y_av, te_x_eeg, te_y_eeg)
