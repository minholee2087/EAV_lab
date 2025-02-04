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

class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, num_layers=2):
        super(ShallowConvNet, self).__init__()

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
        #self.fc = nn.Linear(2600, 768, bias=False)
        self.fc = nn.Linear(2600, nb_classes)

    def feature(self, x):
        """Extract features before the final classification layer."""
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
        flattened_x = torch.flatten(x, 1)  # Feature output
        return flattened_x

    def forward(self, x):
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

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

class ShallowConvNet1(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, num_layers=2):
        super(ShallowConvNet1, self).__init__()

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
        #self.fc = nn.Linear(2600, 768, bias=False)
        self.fc = nn.Linear(2600, nb_classes)

    def feature(self, x):
        """Extract features before the final classification layer."""
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
        flattened_x = torch.flatten(x, 1)  # Feature output
        return flattened_x

    def forward(self, x):
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.fc2(x)
        #x = F.softmax(x, dim=1)

        return x

class ShallowCNN(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5):
        super(ShallowCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 40, (1, 13))
        self.conv2 = nn.Conv2d(40, 40, (Chans, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1)

        # Pooling and dropout
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(dropoutRate)

        # Fully connected layer
        self.fc = nn.Linear(40 * 1 * 65, 256)
        self.fc1 = nn.Linear(256, 2)
        self.fc2 = nn.Linear(2, nb_classes)


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


    def feature(self, x):
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
        x = self.fc1(x)

        return x


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
        x = self.fc1(x)
        x = self.fc2(x)

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

class MultiTaskShallowConvNet_upd(nn.Module):
    def __init__(self, nb_classes_emotion=5, nb_classes_arousal=2, nb_classes_valence=2, Chans=30, Samples=500,
                 dropoutRate=0.5, num_layers=2):
        super(MultiTaskShallowConvNet_upd, self).__init__()

        self.conv1_depth = 40
        self.eeg_ch = 30

        # Convolutional block
        self.conv1 = nn.Conv2d(1, self.conv1_depth, (1, 13), bias=False)
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(dropoutRate)

        # Added ReLU activation after the convolution layer for non-linearity
        self.relu1 = nn.ReLU()

        # Transformer Parameters
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

        # Fully connected layers for multi-task
        self.fc = nn.Linear(2600, nb_classes_emotion)
        self.fc2 = nn.Linear(2600, nb_classes_arousal)
        self.fc3 = nn.Linear(2600, nb_classes_valence)

        # Added ReLU activation before the output layer for each task
        self.relu2 = nn.ReLU()

    def feature(self, x):
        """Extract shared features before task-specific classification layers."""
        x = self.conv1(x)

        # Added ReLU activation after convolution layer
        x = self.relu1(x)

        batch, channels, height, width = x.shape

        # Apply transformer
        V = self.embedding(x)
        for layer in self.transformer_layers:
            V = layer(V)

        # Batchnorm after transformer
        x = V.permute(0, 2, 1).unsqueeze(2)
        x = self.batchnorm(x)

        # Optional: Applying ReLU again after batchnorm
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

        # Apply ReLU activation before final fully connected layers
        x = self.relu2(x)

        # Task-specific outputs
        emotion_output = self.fc(x)
        arousal_output = self.fc2(x)
        valence_output = self.fc3(x)

        return emotion_output, arousal_output, valence_output

class EEGNet_elu(nn.Module):
    def __init__(self, nb_classes_emotion=5, nb_classes_arousal=2, nb_classes_valence=2, Chans=30, Samples=500, dropoutRate=0.5):
        super(EEGNet_elu, self).__init__()

        # Temporal Convolution
        self.temporal_conv = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)  # Match input size
        self.temporal_bn = nn.BatchNorm2d(16)

        # Depthwise Convolution
        self.depthwise_conv = nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(32)

        # Separable Convolution
        self.separable_conv = nn.Conv2d(32, 64, (1, 16), padding=(0, 8), bias=False)
        self.separable_bn = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.AvgPool2d((1, 8))  # Adjust pooling to ensure reduced spatial dimensions

        self.dropout = nn.Dropout(dropoutRate)

        # Fully connected layers
        self.fc = nn.Linear(448, nb_classes_emotion)  # Final flattened size is (64 * 57)
        #self.fc = nn.Linear(flatten_dim, nb_classes_emotion)
        self.fc2 = nn.Linear(448, nb_classes_arousal)
        self.fc3 = nn.Linear(448, nb_classes_valence)


    def feature(self, x):
        """Extract shared features before task-specific classification layers."""
        # Temporal Convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        #x = torch.square(x)  # Squaring as per original EEGNet
        x = torch.nn.functional.elu(x)

        # Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = torch.nn.functional.elu(x)
        #x = torch.square(x)  # Squaring
        x = self.pool(x)

        # Separable Convolution
        x = self.separable_conv(x)
        x = self.separable_bn(x)
        #x = torch.square(x)  # Squaring
        x = torch.nn.functional.elu(x)
        x = self.pool(x)

        # Flatten
        x = x.squeeze(2)  # remove the height dimension? as in shallow+transf
        x = self.dropout(x)
        flattened_x = torch.flatten(x, 1)
        return flattened_x

    def forward(self, x):
        """Forward pass through EEGNet."""
        x = self.feature(x)
        emotion_output = self.fc(x)
        arousal_output = self.fc2(x)
        valence_output = self.fc3(x)

        return emotion_output, arousal_output, valence_output

class EEGNet_upd(nn.Module):
    def __init__(self, nb_classes_emotion=5, nb_classes_arousal=2, nb_classes_valence=2, Chans=30, Samples=500, dropoutRate=0.5):
        super(EEGNet_upd, self).__init__()

        # Temporal Convolution
        self.temporal_conv = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)  # Match input size
        self.temporal_bn = nn.BatchNorm2d(16)

        # Depthwise Convolution
        self.depthwise_conv = nn.Conv2d(16, 32, (Chans, 1), groups=16, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(32)

        # Separable Convolution
        self.separable_conv = nn.Conv2d(32, 64, (1, 16), padding=(0, 8), bias=False)
        self.separable_bn = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.AvgPool2d((1, 8))  # Adjust pooling to ensure reduced spatial dimensions

        self.dropout = nn.Dropout(dropoutRate)

        # Fully connected layers
        self.fc = nn.Linear(448, nb_classes_emotion)  # Final flattened size is (64 * 57)
        self.fc2 = nn.Linear(448, nb_classes_arousal)
        self.fc3 = nn.Linear(448, nb_classes_valence)

    def feature(self, x):
        """Extract shared features before task-specific classification layers."""
        # Temporal Convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = torch.nn.functional.relu(x)  # ELU

        # Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = torch.nn.functional.relu(x)  # ELU
        x = self.pool(x)

        # Separable Convolution
        x = self.separable_conv(x)
        x = self.separable_bn(x)
        x = torch.nn.functional.relu(x)  # ReLU instead of ELU
        x = self.pool(x)

        # Flatten
        x = x.squeeze(2)  # remove the height dimension?
        x = self.dropout(x)
        flattened_x = torch.flatten(x, 1)

        return flattened_x

    def forward(self, x):
        """Forward pass through EEGNet."""
        x = self.feature(x)
        emotion_output = self.fc(x)
        arousal_output = self.fc2(x)
        valence_output = self.fc3(x)

        return emotion_output, arousal_output, valence_output

class OrigShallowConvNet(nn.Module):
    def __init__(self, nb_classes_emotion=5, nb_classes_arousal=2, nb_classes_valence=2, Chans=30, Samples=500, dropoutRate=0.5):
        super(OrigShallowConvNet, self).__init__()

        # First convolutional layer
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), stride=(1, 1), bias=False),  # Temporal convolution
            nn.Conv2d(40, 40, kernel_size=(Chans, 1), stride=(1, 1), bias=False),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))  # Pooling (Downsampling)
        )

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropoutRate)

        # Fully connected layers for multitask outputs
        flattened_size = self.calculate_feature_dim(Chans, Samples)
        self.fc = nn.Linear(flattened_size, nb_classes_emotion)
        self.fc2 = nn.Linear(flattened_size, nb_classes_arousal)
        self.fc3 = nn.Linear(flattened_size, nb_classes_valence)

    def calculate_feature_dim(self, chans, samples):
        """Calculate the feature dimensionality after the first convolutional layer."""
        dummy_input = torch.zeros(1, 1, chans, samples)  # Simulate an input batch
        x = self.firstconv(dummy_input)
        return x.shape[1] * x.shape[2] * x.shape[3]  # Channels x Height x Width

    def feature(self, x):
        """Extract shared features before the fully connected layers."""
        x = self.firstconv(x)
        x = torch.pow(x, 2)  # Square the activations
        x = torch.log(torch.clamp(x, min=1e-6))  # Apply logarithmic scaling
        x = x.squeeze(2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # Flatten the features
        return x

    def forward(self, x):
        # Extract shared features
        x = self.feature(x)

        # Task-specific outputs
        emotion_output = self.fc(x)
        arousal_output = self.fc2(x)
        valence_output = self.fc3(x)

        return emotion_output, arousal_output, valence_output

class MultiTaskTCT(nn.Module):
    def __init__(self, nb_classes_emotion=5, nb_classes_arousal=2, nb_classes_valence=2, Chans=30, Samples=500,
                 dropoutRate=0.5, d_model=64, nhead=4, num_layers=2):
        super(MultiTaskTCT, self).__init__()

        # Temporal convolutional block (similar to firstconv)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), stride=(1, 1), bias=False),  # Temporal convolution
            nn.Conv2d(40, 40, kernel_size=(Chans, 1), stride=(1, 1), bias=False),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.ReLU(),  # Activation
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))  # Pooling
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropoutRate)

        # Transformer encoder block
        self.d_model = d_model
        self.fc_projection = nn.Linear(40, d_model)  # Map to d_model dimension for Transformer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropoutRate, batch_first=True),
            num_layers=num_layers
        )

        # Fully connected layers for multitask outputs
        flattened_size = self.calculate_feature_dim(Chans, Samples)
        self.fc = nn.Linear(flattened_size, nb_classes_emotion)
        self.fc2 = nn.Linear(flattened_size, nb_classes_arousal)
        self.fc3 = nn.Linear(flattened_size, nb_classes_valence)

    def calculate_feature_dim(self, chans, samples):
        """Calculate the feature dimensionality after the transformer."""
        dummy_input = torch.zeros(1, 1, chans, samples)
        x = self.temporal_conv(dummy_input)  # Apply temporal convolution
        x = x.squeeze(2).permute(0, 2, 1)  # Prepare for Transformer
        x = self.fc_projection(x)  # Project to d_model
        x = self.transformer_encoder(x)  # Apply Transformer
        return x.shape[1] * x.shape[2]  # Flattened feature size

    def feature(self, x):
        """Extract shared features before the fully connected layers."""
        # Temporal convolution
        x = self.temporal_conv(x)  # [batch, 40, 1, reduced_samples]
        x = x.squeeze(2).permute(0, 2, 1)  # [batch, reduced_samples, 40]

        # Project to match d_model
        x = self.fc_projection(x)  # [batch, reduced_samples, d_model]

        # Apply Transformer
        x = self.transformer_encoder(x)  # [batch, reduced_samples, d_model]

        # Flatten features
        x = torch.flatten(x, 1)  # [batch, features]
        return x

    def forward(self, x):
        # Extract shared features
        x = self.feature(x)

        # Task-specific outputs
        emotion_output = self.fc(x)
        arousal_output = self.fc2(x)
        valence_output = self.fc3(x)

        return emotion_output, arousal_output, valence_output

class ConformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(ConformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            #nn.ReLU(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multihead self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Feedforward layer
        ff_output = self.ff(x)
        x = self.layer_norm2(x + ff_output)
        return x

class MultiTaskEEGConformer(nn.Module):
    def __init__(self,
                 nb_classes_emotion=5,
                 nb_classes_arousal=2,
                 nb_classes_valence=2,
                 chans=30,
                 samples=500,
                 embed_dim=64,
                 num_heads=4,
                 ff_dim=128,
                 num_layers=2,
                 dropout=0.5):
        super(MultiTaskEEGConformer, self).__init__()

        # Convolutional feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25), stride=(1, 1), bias=False)
        #self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, embed_dim, kernel_size=(chans, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))  # Pooling
        self.dropout = nn.Dropout(dropout)

        # Conformer layers
        self.conformer_layers = nn.ModuleList([
            ConformerLayer(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Dynamically determine the size of the flattened feature vector
        self.flattened_size = embed_dim * (samples - 24)

        # Task-specific classifiers
        self.fc = nn.Linear(self.flattened_size, nb_classes_emotion)
        self.fc2 = nn.Linear(self.flattened_size, nb_classes_arousal)
        self.fc3 = nn.Linear(self.flattened_size, nb_classes_valence)

    def feature(self, x):
        """Extract shared features from the input."""
        # Input shape: [batch, 1, chans, samples]
        batch_size = x.size(0)

        # Convolutional feature extraction
        x = self.conv1(x)  # [batch, 64, chans, samples - 24]
        #x = self.batchnorm1(x)
        #x = F.elu(x)

        x = self.conv2(x)  # [batch, embed_dim, 1, samples - 24]
        x = self.batchnorm(x)
        x = F.elu(x)
        x = self.activation(x)
        x = x.squeeze(2)  # [batch, embed_dim, samples - 24]

        x = x.permute(2, 0, 1)  # [samples - 24, batch, embed_dim]

        # Pass through Conformer layers
        for layer in self.conformer_layers:
            x = layer(x)  # [samples - 24, batch, embed_dim]

        x = x.permute(1, 0, 2)  # [batch, samples - 24, embed_dim]
        x = x.flatten(1)  # [batch, embed_dim * (samples - 24)]

        return x

    def forward(self, x):
        # Extract shared features
        features = self.feature(x)

        # Task-specific outputs
        emotion_output = self.fc(features)
        arousal_output = self.fc2(features)
        valence_output = self.fc3(features)

        return emotion_output, arousal_output, valence_output

class Trainer_eeg:
    def __init__(self, model, data, lr=1e-3, batch_size=280, num_epochs=100, device=None):

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.tr_x, self.tr_y, self.te_x, self.te_y = data
        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self, type_m = None):
        self.model.train()  # Set model to training mode
        for epoch in range(self.num_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if type_m == 'cnn':
                    with torch.no_grad():
                        self.model.module.fc.weight.data = torch.renorm(
                            self.model.module.fc.weight.data, p=2, dim=0, maxnorm=0.5
                        )

                #if batch_idx % 100 == 0:
                    #print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}")

            if self.test_dataloader:
                self.validate()

    def validate(self):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for data, targets in self.test_dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(data)
                loss = self.criterion(scores, targets)
                total_loss += loss.item()
                predictions = scores.argmax(dim=1)
                total_correct += (predictions == targets).sum().item()

        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / len(self.test_dataloader.dataset)
        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

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
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_emotion, dtype=torch.long),
            torch.tensor(y_arousal, dtype=torch.long),
            torch.tensor(y_valence, dtype=torch.long),
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
                    self.model.module.fc.weight.data = torch.renorm(self.model.module.fc.weight.data, p=2, dim=0,
                                                                    maxnorm=0.5)
                    self.model.module.fc2.weight.data = torch.renorm(self.model.module.fc2.weight.data, p=2, dim=0,
                                                                     maxnorm=0.5)
                    self.model.module.fc3.weight.data = torch.renorm(self.model.module.fc3.weight.data, p=2, dim=0,
                                                                     maxnorm=0.5)

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


#if __name__ == "__main__":
#    model = ShallowConvNet(nb_classes=5, Chans=30, Samples=500, num_layers=6)
