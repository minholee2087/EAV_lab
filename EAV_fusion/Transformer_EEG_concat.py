import torch 
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
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
        
        self.midsample = nn.Linear(40, 256)
        self.endsample = nn.Linear(256*3, 40)

    def forward(self, x):
        # Attention and residual connection
        attn_output = self.attention(x)
        x = x + self.dropout(self.norm1(attn_output))

        # Feed-forward and residual connection
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(self.norm2(ffn_output))

        return x


class EEG_Encoder(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, num_layers=12):
        super(EEG_Encoder, self).__init__()

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
        self.num_layers=num_layers
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim=embed_dim, num_heads=self.num_heads, qkv_dim=self.qkv_dim, drop_p=dropoutRate)
            for _ in range(num_layers)
        ])

        self.batchnorm = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1)
        self.fc = nn.Linear(2600, 5, bias=False)

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
        x = F.softmax(x, dim=1)

        return x
    def forward_ending(self, x):#we take the feature map of the eeg encoded after 4 layers and put it in
        
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.batchnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7, max=10000))

        x = x.squeeze(2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

            
