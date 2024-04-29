import torch
import torch.nn as nn
import numpy as np
from torch import FloatTensor


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_size = 15
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.first_layer = nn.Linear(self.feature_size,2*self.feature_size)
        self.first_layer.weight = nn.Parameter(FloatTensor(np.ones((2*self.feature_size,self.feature_size))/self.feature_size))
        self.first_layer.bias.data.zero_()
        
        self.first_activ = nn.SELU()
        
        self.second_layer = nn.Linear(self.feature_size*2,self.feature_size*2)
        self.second_layer.weight = nn.Parameter(FloatTensor(np.ones((2*self.feature_size,2*self.feature_size))/self.feature_size))
        self.second_layer.bias.data.zero_()
        
        self.second_activ = nn.SELU()
        
        self.third_layer = nn.Linear(self.feature_size*2,self.feature_size*2)
        self.third_layer.weight = nn.Parameter(FloatTensor(np.ones((2*self.feature_size,2*self.feature_size))/self.feature_size))
        self.third_layer.bias.data.zero_()
        
        self.third_activ = nn.SELU()
        
        self.fourth_layer = nn.Linear(self.feature_size*2,self.feature_size)
        self.fourth_layer.weight = nn.Parameter(FloatTensor(np.ones((self.feature_size,2*self.feature_size))/self.feature_size))
        self.fourth_layer.bias.data.zero_()
        
        self.output_layer = nn.Linear(self.feature_size,1)
        self.output_layer.weight = nn.Parameter(FloatTensor(np.ones((1,self.feature_size))/self.feature_size))
        self.output_layer.bias.data.zero_()
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = x.to(self.device)
        first = self.first_layer(x)
        first_act = self.first_activ(first)
        
        second = self.second_layer(first_act)
        second_act = self.second_activ(second)
        
        third = self.third_layer(second_act)
        third_act = self.third_activ(third)
        
        fourth = self.fourth_layer(third_act)
        
        output = self.output_layer(fourth)
        
        result = self.sigmoid(output)
        
#         class_result = torch.where(result >= 0.5, 1, 0)
        return result