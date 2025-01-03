import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerPrintComponent(nn.Module):
    r"""
    MLP for molecular fingerprint feature learning
    
    我们通过学习三种不同的分子指纹来提取分子的特征：
    1. the MACCS fingerprint
    2. the Pharmacophore ErG fingerprint
    3. the PubChem fingerprint
    
    将三种不同的分子指纹连接在一起，然后通过一个 MLP 来提取特征。
    """
    def __init__(
        self, 
        in_channels: int = 1489, 
        mid_channels: int = 512,
        out_channels: int = 96
    ):
        super(FingerPrintComponent, self).__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
        
class SqueezeExcitationBlock(nn.Module):
    r"""
    The Squeeze-and-Excitation block (SE block)     
    
    - Squeece Operation: Global Average Pooling
    - Excitation Operation: MLP with two fully connected layers
    """
    def __init__(
        self, 
        in_channels
    ):
        super(SqueezeExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels, in_channels // 16, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels // 16, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        
        return x
    

class MultiScaleCNNSE(nn.Module):
    r"""
    Multiscale CNN-SE for SMILES feature learning
    
    多尺度 CNN-SE SMILES 学习组件将 SMILES 嵌入输入到不同核大小的卷积层中，
    用于提取不同尺度的特征，然后将提取的特征输入到相应的 SE 块中。最后，将来自不同尺度 CNN-SE 的特征进行串联
    """
    def __init__(
        self, 
        embedding_dim: int = 128, 
        filter_channels: int = 32,
        out_channels: int = 96, 
    ):
        super(MultiScaleCNNSE, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        self.embedding = nn.Embedding(100, embedding_dim)
        # 卷积核大小为 2
        self.conv1 = nn.Conv1d(100, filter_channels, 2)
        self.fc1 = nn.Linear(32 * 127, out_channels)
        # 卷积核大小为 4
        self.conv2 = nn.Conv1d(100, filter_channels, 4)
        self.fc2 = nn.Linear(32 * 125, out_channels)
        # 卷积核大小为 8
        self.conv3 = nn.Conv1d(100, filter_channels, 8)
        self.fc3 = nn.Linear(32 * 121, out_channels)
        # SE block
        self.se = SqueezeExcitationBlock(filter_channels)
        self.out = nn.Linear(288, out_channels)
        
    def forward(self, x):
        # 首先把 smiles vector 嵌入到 128 维的空间中
        x = self.embedding(x)
        # 卷积核大小为 2
        y1 = self.conv1(self.relu(x))
        se1 = self.se(x)
        y1 = y1 * se1
        # 卷积核大小为 4
        y2 = self.conv2(self.relu(x))
        se2 = self.se(x)
        y2 = y2 * se2
        # 卷积核大小为 8
        y3 = self.conv3(self.relu(x))
        se3 = self.se(x)
        y3 = y3 * se3
        
        # flatten 卷积核为 2 的输出
        y1 = y1.view(-1, 32 * 127)
        y1 = self.fc1(y1)
        # flatten 卷积核为 4 的输出
        y2 = y2.view(-1, 32 * 125)
        y2 = self.fc2(y2)
        # flatten 卷积核为 8 的输出
        y3 = y3.view(-1, 32 * 121)
        y3 = self.fc3(y3)
        # concat
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.out(y)
        
        return y
    
    
