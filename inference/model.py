# Build model structure
import torch
from transformers import get_cosine_schedule_with_warmup
from torch import optim, nn
from torch.nn import functional as F

class ArcFace(nn.Module):
    
    def __init__(self,in_features,out_features, margin = 0.5 ,scale = 20):
        super(ArcFace, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(out_features,in_features))
        nn.init.xavier_normal_(self.weights)
        
    def forward(self,features,targets):
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weights), bias=None) 
        cos_theta = cos_theta.clip(-1, 1)
        
        arc_cos = torch.acos(cos_theta)
        M = F.one_hot(targets, num_classes = self.out_features) * self.margin
        arc_cos = arc_cos + M
        
        cos_theta_2 = torch.cos(arc_cos)
        logits = cos_theta_2 * self.scale
        return logits

class Model(torch.nn.Module):
    def __init__(self, vit_model, feature_extractor, num_labels):

        super(Model, self).__init__()
        self.vit = vit_model
        self.feature_extractor = feature_extractor
        self.num_labels = num_labels
        self.arcface = ArcFace(768, num_labels, margin=0.3)
    
    def forward(self, batch, device):
        outputs = self.vit(batch['image'].to(device))
        last_hidden_state = outputs.last_hidden_state[:,0,:]
        if batch['mode'] == 'train':
            return self.arcface(last_hidden_state)
        else:
            return F.linear(F.normalize(last_hidden_state), F.normalize(self.arcface.weights))

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