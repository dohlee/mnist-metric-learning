import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedHead(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(FullyConnectedHead, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, features):
        return self.fc(features)


class ArcFaceHead(nn.Module):

    def __init__(self, dim_in, dim_out, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceHead, self).__init__()
        
        self.dim_in, self.dim_out = dim_in, dim_out
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(dim_out, dim_in))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, features, target=None):
        cosine_theta = F.linear(F.normalize(features), F.normalize(self.weight))  # NxW
        if not self.training:
            return cosine_theta * self.s

        sine_theta = torch.sqrt((1.0 - torch.pow(cosine_theta, 2)).clamp(0, 1))
        
        # Derived from:
        # phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine_theta * self.cos_m - sine_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine_theta > 0, phi, cosine_theta)
        else:
            phi = torch.where(cosine_theta > self.th, phi, cosine_theta - self.mm)

        one_hot = torch.zeros(cosine_theta.size())
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_theta)
        output *= self.s

        return output

    def summary(self):
        print(self.training)

class CosFaceHead(nn.Module):

    def __init__(self):
        super(CosFaceHead, self).__init__()
        pass

if __name__ == '__main__':
    arc = ArcFaceHead(dim_in=10, dim_out=100)

    arc.train()
    arc.summary()

    arc.eval()
    arc.summary()
