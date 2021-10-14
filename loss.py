import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """Some Information about MyModule"""
    def __init__(self,margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1,output_2,label):
        euclidean_dist = F.pairwise_distance(output_1,output_2,keepdim = True)
        loss = torch.mean((1-label)*torch.pow(euclidean_dist, 2) + (label)*torch.pow(torch.clamp(self.margin-euclidean_dist, min=0.0),2))
        return loss