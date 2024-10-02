import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, input, target):
        loss = F.mse_loss(input, target)
        return loss

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(b * c * h * w) # normalize

    def forward(self, input, target):
        loss = F.mse_loss(self.gram_matrix(input), self.gram_matrix(target))
        return loss
