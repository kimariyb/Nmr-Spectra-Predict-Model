import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.batchnorm as batchnorm
import torch_geometric.nn as geo_nn


class NodeLevelBatchNorm(batchnorm._BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    
    Shape
    ----------    
    Input: [batch_nodes_dim, node_feature_dim]
    Output: [batch_nodes_dim, node_feature_dim]
    
    Parameters
    ----------
    batch_nodes_dim:
        all nodes of a batch graph
    """
    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        momentum: float = 0.1, 
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)
               
               
class GraphConvBatchNorm(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels
    ):
        super(GraphConvBatchNorm, self).__init__()
        self.conv = geo_nn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        return x
    
# TODO: Implement the DenseLayer class
class DenseLayer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels
    ):
        super(DenseLayer, self).__init__()
        self.conv = GraphConvBatchNorm(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        
        return x
    
# TODO: Implement the DenseBlock class
class DenseBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_layers
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels, out_channels)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            
        return x
    

# TODO: Implement the GraphDenseNet class
class GraphDenseNet(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_layers, 
        num_blocks
    ):
        super(GraphDenseNet, self).__init__()
        self.blocks = nn.ModuleList([
            DenseBlock(in_channels, out_channels, num_layers)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, edge_index):
        for block in self.blocks:
            x = block(x, edge_index)
            
        return x