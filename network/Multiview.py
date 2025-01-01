import torch
import torch.nn as nn

from network.attention import CrossAttentionBlock


class MultiViewBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
    ):
        super(MultiViewBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        
        return x


class MultiViewNet(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
    ):
        super(MultiViewNet, self).__init__()

        self.encoder = CrossAttentionBlock(in_channels, out_channels)
        self.out_channels = out_channels
        
        self.classifier = nn.Sequential(
            MultiViewBlock(in_channels, 1024),
            MultiViewBlock(1024, 1024),
            MultiViewBlock(1024, 256),
            nn.Linear(256, out_channels)
        )
        
    def forward(self, fp, smi, graph):
        graph_x, fp_x2, smi_x, fp_x1 = self.encoder(fp, smi, graph)
        x = torch.cat([graph_x, fp_x2, smi_x, fp_x1], dim=1)
        x = self.classifier(x)
        
        return x
        