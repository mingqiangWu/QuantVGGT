import torch
import torch.nn as nn
import torch.nn.functional as F
import deploy
    
class DeployLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super(DeployLinear, self).__init__()
        self.linear = deploy.nn.Linear4bit.from_float(linear)
        self.quantizer = deploy.nn.Quantizer()
        # self.inp_trans = deploy.nn.OnlineTrans(linear.weight.shape[1], trans="matmul")
        self.inp_trans = deploy.nn.OnlineTrans(linear.weight.shape[1], trans="had")

    def forward(self, x):
        x = self.inp_trans(x)
        x = self.quantizer(x)
        return self.linear(x)