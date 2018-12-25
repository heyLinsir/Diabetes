import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """docstring for MLP"""
    def __init__(self, input_size, class_num=3):
        super(MLP, self).__init__()
        print('input size: %d' % (input_size))
        self.linear = nn.Sequential(nn.Linear(input_size, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, class_num))

    def forward(self, x, is_training=True):
        x = F.dropout(x, p=0.5, training=is_training)
        prob = self.linear(x)
        prediction = torch.max(prob, dim=1)[1]
        return prob, prediction
