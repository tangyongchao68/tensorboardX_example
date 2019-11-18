import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

x = torch.randn(10, 16, 30, 32, 34)
# batch, channel , height , width
print(x.shape)


class Net_1D(nn.Module):
    def __init__(self):
        super(Net_1D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(3, 2, 2), stride=(2, 2, 1), padding=[2,2,2]),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.layers(x)
        log_probs = F.log_softmax(output, dim=1)
        return  log_probs


n = Net_1D()  # in_channel,out_channel,kennel,
print(n)
y = n(x)
print(y.shape)

with SummaryWriter(log_dir='./log', comment='Net1')as w:
    w.add_graph(n, x)
