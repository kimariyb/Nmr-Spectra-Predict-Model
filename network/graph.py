import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.batchnorm as batchnorm
import torch_geometric.nn as geo_nn

from collections import OrderedDict


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
    

class DenseLayer(nn.Module):
    def __init__(
        self, 
        num_input_features: int, 
        growth_rate: int = 32,
        batch_size: int = 4
    ):
        super(DenseLayer, self).__init__()
        self.conv1 = GraphConvBatchNorm(num_input_features, int(growth_rate * batch_size))
        self.conv2 = GraphConvBatchNorm(int(growth_rate * batch_size), growth_rate)
        
    def batch_func(self, x):
        concat_features = torch.cat(x, 1)
        concat_features = self.conv1(concat_features)
        
        return concat_features
        
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        
        x = self.batch_func(x)
        x = self.conv2(x)
        
        return x
    

class DenseBlock(nn.ModuleDict):
    def __init__(
        self, 
        num_layers: int, 
        num_input_features: int, 
        growth_rate: int = 32,
        batch_size: int = 4
    ):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, batch_size)
            self.add_module('layer%d' % (i + 1), layer)
        
    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            x = layer(x)
            features.append(x)
            x = features
            
        x = torch.cat(x, 1)
                        
        return x
    

class GraphDenseNet(nn.Module):
    def __init__(
        self, 
        num_input_features: int = 32, 
        growth_rate: int = 32, 
        block_config: tuple = (3, 3, 3, 3),
        batch_sizes: list = [2, 3, 4, 4],
        out_channels: int = 1
    ):
        super(GraphDenseNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                'conv0', GraphConvBatchNorm(num_input_features, 32)
            ])
        )
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate, batch_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)
            
            bn = GraphConvBatchNorm(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), bn)
            num_input_features = num_input_features // 2
            
        self.classifer = nn.Linear(num_input_features, out_channels)
            
        
    def forward(self, x, edge_index):
        x = self.features(x, edge_index)
        x = geo_nn.global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long))
        x = self.classifer(x)
            
        return x