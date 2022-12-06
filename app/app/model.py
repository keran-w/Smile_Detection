# Build model structure
import torch
from torch import nn
from torch.nn import functional as F


class ArcFace(nn.Module):

    def __init__(self, in_features, out_features, margin=0.5, scale=20):
        super(ArcFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weights)

    def forward(self, features, targets):
        cos_theta = F.linear(F.normalize(features),
                             F.normalize(self.weights), bias=None)
        cos_theta = cos_theta.clip(-1, 1)

        arc_cos = torch.acos(cos_theta)
        M = F.one_hot(targets, num_classes=self.out_features) * self.margin
        arc_cos = arc_cos + M

        cos_theta_2 = torch.cos(arc_cos)
        logits = cos_theta_2 * self.scale
        return logits


class VitModel(torch.nn.Module):
    def __init__(self, vit_model, feature_extractor, num_labels):

        super(VitModel, self).__init__()
        self.vit = vit_model
        self.feature_extractor = feature_extractor
        self.num_labels = num_labels
        self.arcface = ArcFace(768, num_labels, margin=0.3)

    def forward(self, batch, device):
        outputs = self.vit(batch['image'].to(device))
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        if batch['mode'] == 'train':
            return self.arcface(last_hidden_state)
        else:
            return F.linear(F.normalize(last_hidden_state), F.normalize(self.arcface.weights))


class BasicCNNClassifier(nn.Module):
    def __init__(self, num_classes=2, device='cpu'):
        super(BasicCNNClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.LazyConv2d(
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=2,
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.LazyConv2d(32, 5, 1, 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.out = nn.LazyLinear(num_classes)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.conv1(x.to(self.device))
        x = self.conv2(x)
        logits = x
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class VGG16(nn.Module):
    '''https://blog.paperspace.com/vgg-from-scratch-pytorch/'''

    def __init__(self, num_classes=2, device='cpu'):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyLinear(4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        self.device = device
        self.to(device)

    def forward(self, x):
        out = self.layer1(x.to(self.device))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class FocalLoss(nn.Module):
    'Focal Loss - https://arxiv.org/abs/1708.02002'

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
