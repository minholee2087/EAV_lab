import sys
import os

# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.insert(0, parent_dir)

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim):
        super(PatchEmbedding, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.W_v = nn.ModuleList([nn.Linear(30, 1, bias=False) for _ in range(40)])

    def forward(self, x):
        outputs_V_res = []
        for i in range(40):
            x_head = x[:, i, :, :].permute(0, 2, 1)  # (batch, head_dim, seq_len)
            V = self.W_v[i](x_head)
            outputs_V_res.append(V)
        outputs_V_res = torch.cat(outputs_V_res, dim=-1)
        return outputs_V_res

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim

        self.W_q = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.W_v = nn.Linear(self.head_dim, self.qkv_dim, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        outputs = []
        outputs_V_res = []
        for i in range(self.num_heads):
            x_head = x[:, i, :, :]
            Q = self.W_q(x_head)
            K = self.W_k(x_head)
            V = self.W_v(x_head)

            Q = Q.permute(0, 2, 1)
            attn_scores = torch.matmul(Q, K) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)

            attn_output = torch.matmul(V, attn_weights)
            outputs.append(attn_output)
            outputs_V_res.append(V)

        attn_output = torch.cat(outputs, dim=-1)
        outputs_V_res = torch.cat(outputs_V_res, dim=-1)

        return x_head + outputs_V_res  # residual

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, expansion=4, drop_p=0.5):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion)
        self.fc2 = nn.Linear(embed_dim * expansion, embed_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim, expansion=4, drop_p=0.5):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, qkv_dim)
        self.feed_forward = FeedForwardBlock(embed_dim, expansion, drop_p)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        # Attention and residual connection
        attn_output = self.attention(x)
        x = x + self.dropout(self.norm1(attn_output))

        # Feed-forward and residual connection
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(self.norm2(ffn_output))

        return x

class MultiTaskShallowConvNet(nn.Module):
    def __init__(self, nb_classes_emotion = 5, nb_classes_arousal = 2, nb_classes_valence = 2, Chans=30, Samples=500, dropoutRate=0.5, num_layers=2):
        super(MultiTaskShallowConvNet, self).__init__()

        self.conv1_depth = 40
        self.eeg_ch = 30

        # Convolutional block
        self.conv1 = nn.Conv2d(1, self.conv1_depth, (1, 13), bias=False)
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(dropoutRate)

        self.qkv_dim = 40
        self.num_heads = 1
        embed_dim = 40
        self.embedding = PatchEmbedding(embed_dim, self.num_heads, self.qkv_dim)

        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim=embed_dim, num_heads=self.num_heads, qkv_dim=self.qkv_dim, drop_p=dropoutRate)
            for _ in range(num_layers)
        ])

        self.batchnorm = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1)

        # Shared fully connected layer
        #self.fc = nn.Linear(2600, 256)
        self.fc = nn.Linear(2600, 5)
        self.fc2 = nn.Linear(2600, 2)
        self.fc3 = nn.Linear(2600, 2)

    def feature(self, x):
        """Extract shared features before task-specific classification layers."""
        x = self.conv1(x)
        batch, channels, height, width = x.shape

        V = self.embedding(x)
        for layer in self.transformer_layers:
            V = layer(V)

        x = V.permute(0, 2, 1).unsqueeze(2)
        x = self.batchnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7, max=10000))

        x = x.squeeze(2)
        x = self.dropout(x)
        flattened_x = torch.flatten(x, 1)  # Shared feature output
        return flattened_x

    def forward(self, x):
        # Extract shared features
        x = self.feature(x)
        # Task-specific outputs
        emotion_output = self.fc(x)
        arousal_output = self.fc2(x)
        valence_output = self.fc3(x)

        return emotion_output, arousal_output, valence_output

class Trainer_eeg_multitask:
    def __init__(self, model, data, lr=1e-3, batch_size=280, num_epochs=100, device=None, multiloss = True):

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Unpack data
        (self.tr_x, self.tr_y_emotion, self.tr_y_arousal, self.tr_y_valence,
         self.te_x, self.te_y_emotion, self.te_y_arousal, self.te_y_valence) = data

        # Prepare dataloaders
        self.train_dataloader = self._prepare_dataloader(
            self.tr_x, self.tr_y_emotion, self.tr_y_arousal, self.tr_y_valence, shuffle=True
        )
        self.test_dataloader = self._prepare_dataloader(
            self.te_x, self.te_y_emotion, self.te_y_arousal, self.te_y_valence, shuffle=False
        )

        self.model = model
        self.criterion_emotion = nn.CrossEntropyLoss()
        self.criterion_arousal = nn.CrossEntropyLoss()
        self.criterion_valence = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.multiloss= multiloss

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def _prepare_dataloader(self, x, y_emotion, y_arousal, y_valence, shuffle=False):
        dataset = TensorDataset(
            #torch.tensor(x, dtype=torch.float32),
            torch.clone(x).detach(),
            #torch.tensor(y_emotion, dtype=torch.long),
            torch.clone(y_emotion).detach(),
            #torch.tensor(y_arousal, dtype=torch.long),
            torch.clone(y_arousal).detach(),
            #torch.tensor(y_valence, dtype=torch.long),
            torch.clone(y_valence).detach(),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self):
        """Train the model across multiple epochs."""
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            total_loss_epoch = 0

            for batch_idx, (data, targets_emotion, targets_arousal, targets_valence) in enumerate(
                    self.train_dataloader):
                # Move data to device
                data = data.to(self.device)
                targets_emotion = targets_emotion.to(self.device)
                targets_arousal = targets_arousal.to(self.device)
                targets_valence = targets_valence.to(self.device)

                # Forward pass
                emotion_output, arousal_output, valence_output = self.model(data)

                # Compute losses
                loss_emotion = self.criterion_emotion(emotion_output, targets_emotion)
                loss_arousal = self.criterion_arousal(arousal_output, targets_arousal)
                loss_valence = self.criterion_valence(valence_output, targets_valence)

                if self.multiloss:
                    if epoch < 30:
                        total_loss = loss_emotion + (0.2 * (loss_arousal + loss_valence))
                    else:
                        total_loss = loss_emotion
                else:
                    total_loss = loss_emotion

                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Apply weight constraints
                with torch.no_grad():
                    model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

                    model_ref.fc.weight.data = torch.renorm(model_ref.fc.weight.data, p=2, dim=0, maxnorm=0.5)
                    model_ref.fc2.weight.data = torch.renorm(model_ref.fc2.weight.data, p=2, dim=0, maxnorm=0.5)
                    model_ref.fc3.weight.data = torch.renorm(model_ref.fc3.weight.data, p=2, dim=0, maxnorm=0.5)

                total_loss_epoch += total_loss.item()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {total_loss_epoch:.4f}")

            # Validate the model at the end of each epoch
            res_emo, res_aro, res_val = self.validate()

        # Return validation results after all epochs are completed
        return res_emo, res_aro, res_val

    def validate(self):
        res_emo = list()
        res_aro = list()
        res_val = list()

        """Evaluate the model on the test dataset."""
        self.model.eval()  # Set model to evaluation mode
        total_correct_emotion = 0
        total_correct_arousal = 0
        total_correct_valence = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets_emotion, targets_arousal, targets_valence in self.test_dataloader:
                # Move data to device
                data = data.to(self.device)
                targets_emotion = targets_emotion.to(self.device)
                targets_arousal = targets_arousal.to(self.device)
                targets_valence = targets_valence.to(self.device)

                # Forward pass
                emotion_output, arousal_output, valence_output = self.model(data)

                # Predictions
                _, predicted_emotion = torch.max(emotion_output, dim=1)
                _, predicted_arousal = torch.max(arousal_output, dim=1)
                _, predicted_valence = torch.max(valence_output, dim=1)

                # Accuracy calculation
                total_correct_emotion += (predicted_emotion == targets_emotion).sum().item()
                total_correct_arousal += (predicted_arousal == targets_arousal).sum().item()
                total_correct_valence += (predicted_valence == targets_valence).sum().item()
                total_samples += targets_emotion.size(0)

        # Calculate accuracies
        emotion_accuracy = total_correct_emotion / total_samples * 100
        arousal_accuracy = total_correct_arousal / total_samples * 100
        valence_accuracy = total_correct_valence / total_samples * 100

        res_emo.append(emotion_accuracy)
        res_aro.append(arousal_accuracy)
        res_val.append(valence_accuracy)

        print(f"Validation - Emotion Accuracy: {emotion_accuracy:.2f}%, Arousal Accuracy: {arousal_accuracy:.2f}%, Valence Accuracy: {valence_accuracy:.2f}%")
        return res_emo, res_aro, res_val
