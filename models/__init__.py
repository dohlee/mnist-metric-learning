import torch
import torch.nn as nn

import const

class MyMNISTBackbone(nn.Module):

    def __init__(self, heads, dim_in, dim_feature):
        super(MyMNISTBackbone, self).__init__()
        assert isinstance(heads, dict)

        self.head_keys = list(heads.keys())
        self.heads = nn.ModuleList(list(heads.values()))
        self.dim_in, self.dim_feature = dim_in, dim_feature 

    def describe(self):
        for param in self.parameters():
            print(param.shape)

        for name, child in self.named_children():
            print(name)

        for head in self.heads:
            print('HEAD:', head)
            for param in head.parameters():
                print(param)

class SimpleDNN(MyMNISTBackbone):

    def __init__(self, heads, dim_in, dim_feature):
        super(SimpleDNN, self).__init__(heads, dim_in, dim_feature)
               
        self.fc1 = nn.Linear(self.dim_in, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, self.dim_feature)
        self.relu = nn.ReLU()

    def forward(self, x, target=None):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)

        out = {'feature': x}
        for head_key, head in zip(self.head_keys, self.heads):
            if self.training and head_key == const.ARCFACE:
                out[head_key] = head(x, target)
            else:
                out[head_key] = head(x)
             
        return out

