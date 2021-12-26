import torch
import torch.nn as nn
from torchvision.models import resnet18


class _Resnet(nn.Module):
  def __init__(self, output_size: int = 5):
    super().__init__()
    self.output_size = output_size

    def init_normal(m):
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  
        m.bias.data.fill_(0.)

    self._resnet = resnet18(pretrained=True)
    self._resnet.fc = nn.Linear(in_features=512, out_features=5, bias=True)
    self._resnet.fc.apply(init_normal)

  def forward(self, data):
    output = self._resnet(data)
    return torch.sigmoid(output)