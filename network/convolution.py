import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerPrintBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1489, 
        mid_channels: int = 512,
        out_channels: int = 96
    ):
        super(FingerPrintBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, mid_channels)
        self.linear2 = nn.Linear(mid_channels, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x
    
        
class SqueezeExcitationBlock(nn.Module):
    def __init__(
        self, 
        in_channels
    ):
        super(SqueezeExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels, in_channels // 16, 1)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv1d(in_channels // 16, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        
        return x
    

class SmilesConvBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim: int = 128, 
        filter_channels: int = 32,
        out_channels: int = 96, 
    ):
        super(SmilesConvBlock, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        
        self.embedding = nn.Embedding(100, embedding_dim)
        
        self.conv1 = nn.Conv1d(100, filter_channels, 2)
        self.linear1 = nn.Linear(32 * 127, out_channels)
        
        self.conv2 = nn.Conv1d(100, filter_channels, 4)
        self.linear2 = nn.Linear(32 * 125, out_channels)
        
        self.conv3 = nn.Conv1d(100, filter_channels, 8)
        self.linear3 = nn.Linear(32 * 121, out_channels)
        
        self.squeeze_excitation = SqueezeExcitationBlock(filter_channels)
        
        self.linear4 = nn.Linear(384, 1024)
        self.linear5 = nn.Linear(1024, 256)
        self.linear6 = nn.Linear(288, out_channels)
        self.out = nn.Linear(256, 1)
        
    def forward(self, x):
        # smiles vector
        x = self.embedding(x)
        # conv1
        conv1 = self.conv1(self.act(x))
        squeeze1 = self.squeeze_excitation(x)
        y1 = conv1 * squeeze1
        # conv 2
        conv2 = self.conv2(self.act(x))
        squeeze2 = self.squeeze_excitation(x)
        y2 = conv2 * squeeze2
        # conv 3
        conv3 = self.conv3(self.act(x))
        squeeze3 = self.squeeze_excitation(x)
        y3 = conv3 * squeeze3
        # flatten
        # linear1
        y1 = y1.view(-1, 32 * 127)
        y1 = self.linear1(y1)
        # linear2
        y2 = y2.view(-1, 32 * 125)
        y2 = self.linear2(y2)
        # linear3
        y3 = y3.view(-1, 32 * 121)
        y3 = self.linear3(y3)
        # concat
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.linear6(y)
        
        return x
    
    
