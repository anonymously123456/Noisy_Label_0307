import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
import torchvision
import numpy as np


class ResNetFc(nn.Module):
    def __init__(self, layers=18):
        super(ResNetFc, self).__init__()
        if layers == 18:
            model_resnet = torchvision.models.resnet18(pretrained=True)
        elif layers == 34:
            model_resnet = torchvision.models.resnet34(pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = model_resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class CIFARModel(nn.Module):
    def __init__(self, class_num=10, layers=18):
        super(CIFARModel, self).__init__()
        self.class_num = class_num
        self.backbone = ResNetFc(layers=layers)
        self.clf = nn.Linear(512, class_num)

    # def forward(self, x, Pi, priors_corr, prior_test):
    def forward(self, x):
        F = self.backbone(x)
        x = self.clf(F)
        g = torch.softmax(x, dim=1)
        # x = self.QfunctionMulticlass(g, Pi, priors_corr)
        return g, F

    def QfunctionMulticlass(self, g, Pi, priors_corr):
        pi_ita = torch.mm(Pi, g.permute(1, 0))
        rou_pi_ita = torch.matmul(priors_corr, pi_ita)
        pi_corr = pi_ita.permute(1, 0) * priors_corr.unsqueeze(0)
        output = (pi_corr.permute(1, 0) / rou_pi_ita).permute(1, 0)
        return output

    def predict(self, x):
        x = self.backbone(x)
        x = self.clf(x)
        g = torch.softmax(x, dim=1)
        return g

    def forward_with_P(self, x, P):
        F = self.backbone(x)
        x = self.clf(F)
        g = torch.softmax(x, dim=1)
        g = g @ P.T
        return g, F

    def forward_bw(self, x):
        x = self.backbone(x)
        x = self.clf(x)
        g = torch.softmax(x, dim=1)
        return g

    def server_forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)
        return x
