import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, emb_size = 128):
        super().__init__()

        self.conv1 = self._conv_layer_set(3 , 32)
        self.conv2 = self._conv_layer_set(32, 64)

        # x = torch.randn(67 ,50).view(1 , 67 , 50)
        self._to_linear = 20
        # self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, emb_size)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def convs(self, x):
        print(x.shape)
        x = F.max_pool3d(F.relu(self.conv1(x)), (2 ,2))
        x = F.max_pool3d(F.relu(self.conv2(x)), (2 ,2))

        if self._to_linear is None:
            s = x[0].shape
            self._to_linear = s[0] * s[1] * s[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim = 1)