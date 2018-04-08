from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np

from process import continuous

class CNNReg(nn.Module):
    def __init__(self, length: int, embedding_dim:int, CUDA:int = 1):
        super(CNNReg, self).__init__()
        self.kernel_size = [2, 3, 4, 5]
        self.channel_out = 10
        self.embedding = nn.Embedding(length, embedding_dim)
        self.conv1 = nn.ModuleList([nn.Conv2d(1, self.channel_out, (k, embedding_dim)) for k in self.kernel_size])
        self.linear1 = nn.Linear(self.channel_out*len(self.kernel_size), 10)
        self.linear2 = nn.Linear(10, 1)
        self.dropout = nn.Dropout()
        self._cuda = CUDA

    def forward(self, data):
        data = np.array(list(map(continuous, data)))
        data = Variable(torch.from_numpy(data).long())
        if self._cuda: data = data.cuda()

        embed = self.embedding(data)
        embed = embed.unsqueeze(1)

        feature_maps = [F.relu(conv(embed)) for conv in self.conv1]
        feature_maps = [feature_map.squeeze(3) for feature_map in feature_maps]

        pooled_output = [F.max_pool1d(feature_map, feature_map.size(2)) for feature_map in feature_maps]
        output = torch.cat(pooled_output, 1)
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)
        return output
