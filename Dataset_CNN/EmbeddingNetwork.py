import torch.nn as nn
import torch
import torchvision.models as models


def set_parameter_requires_grad(model, feature_extracting):
    if (feature_extracting):
        for param in model.parameters():
            param.requires_grad = False


class EmbeddingNetwork(nn.Module):
    def __init__(self, is_pretrained=True, freeze_params=True):
        super(EmbeddingNetwork, self).__init__()

        self.backbone = models.resnet152(pretrained=is_pretrained)
        set_parameter_requires_grad(self.backbone, freeze_params)

        # replace the last classification layer with an embedding layer.
        # num_ftrs = self.backbone.fc.in_features
        # self.backbone.fc = nn.Linear(num_ftrs, emb_dim)
        self.backbone.fc = Identity()

        # make that layer trainable
        if freeze_params:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif not freeze_params:
            for param in self.backbone.parameters():
                param.requires_grad = True

        self.inputsize = 100

    def forward(self, x):

        x = self.backbone(x)
        x = F.normalize(x, p=2.0, dim=1)

        return x


class Identity(torch.nn.Module):
    def init(self):
        super(Identity, self).init()

    def forward(self, x):
        return x