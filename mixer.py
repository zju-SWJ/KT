import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import einops


class TokenMixingMLP(nn.Module): # MLP with input size (B, C, S)
    def __init__(self, in_dim, hidden_dim, norm_dim, p):
        super().__init__()
        self.layernorm = nn.LayerNorm(norm_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout2 = nn.Dropout(p)

    def forward(self, x):
        identity = x
        x = self.layernorm(x)
        x = einops.rearrange(x, 'b s c -> b c s')
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = einops.rearrange(x, 'b c s -> b s c')
        x = x + identity
        return x
    


class ChannelMixingMLP(nn.Module): # MLP with input size (B, S, C)
    def __init__(self, in_dim, hidden_dim, norm_dim, p):
        super().__init__()
        self.layernorm = nn.LayerNorm(norm_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout2 = nn.Dropout(p)

    def forward(self, x):
        identity = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + identity
        return x



class MLPMixer(nn.Module):
    def __init__(self, num_layers=8, S_dim=72, C_dim=128, hidden_S_dim=256, hidden_C_dim=512, dropout=0.):
        super().__init__()
        self.map_c = True if C_dim !=9 else False
        if self.map_c:
            self.map1 = nn.Linear(9, C_dim)
            self.map2 = nn.Linear(C_dim, 9)
        
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"mixer_layer_{i}"] = nn.Sequential(
                TokenMixingMLP(S_dim, hidden_S_dim, C_dim, dropout),
                ChannelMixingMLP(C_dim, hidden_C_dim, C_dim, dropout)
            )
        self.layers = nn.Sequential(layers)

        self.layernorm = nn.LayerNorm(C_dim)
        self.fc1 = nn.Linear(S_dim, hidden_S_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_S_dim, S_dim // 2)

    def forward(self, inputs):
        conv1w, conv2w = inputs
        Batch_1, Out_1, In_1, Kernel1_1, Kernel2_1 = conv1w.shape
        Batch_2, Out_2, In_2, Kernel1_2, Kernel2_2 = conv2w.shape
        assert Batch_1 == Batch_2 and Out_1 == Out_2 and Out_1 == In_1 and Out_1 == In_2 \
            and Kernel1_1 == 3 and Kernel1_2 == 3 and Kernel2_1 == 3 and Kernel2_2 == 3
        
        conv1w, conv2w = einops.rearrange(conv1w, 'b o i s1 s2 -> b (o i) (s1 s2)'), einops.rearrange(conv2w, 'b o i s1 s2 -> b (o i) (s1 s2)')
        x = torch.cat([conv1w, conv2w], dim=1)
        if self.map_c:
            x = self.map1(x)

        x = self.layers(x)

        x = self.layernorm(x)
        x = einops.rearrange(x, 'b s c -> b c s')
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = einops.rearrange(x, 'b c s -> b s c')

        if self.map_c:
            x = self.map2(x) # B, 36, 9
        assert list(x.shape) == [Batch_1, 36, 9]
        
        x = x.reshape(Batch_1, 36, 3, 3) # B, 36, 3, 3

        x1, x2 = torch.chunk(x, 2, dim=1) # B, 18, 3, 3
        assert list(x1.shape) == [Batch_1, 18, 3, 3] and list(x2.shape) == [Batch_2, 18, 3, 3]

        x1, x2 = x1.reshape(Batch_1, 3, 6, 3, 3), x2.reshape(Batch_2, 6, 3, 3, 3)
        return x1, x2