import random
import numpy as np
import torch
import time
import copy
import math
from torch import nn
import model.util_model as util_md


class Model_final(nn.Module):

    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(Model_final, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim

        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )

        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length

        self.clas = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_clas = self.clas(x)
        output_reg = self.reg(x) * self.act(output_clas)
        return output_clas, output_reg


class Model_Mark(nn.Module):

    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(Model_Mark, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim

        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )

        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length

        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_reg = self.reg(x)
        return output_reg


class Model_MTL(nn.Module):

    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(Model_MTL, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim

        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )

        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length

        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

        self.clas = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_reg = self.reg(x)
        output_clas = self.clas(x)
        return output_clas, output_reg



# 2024.01.02. Below are the variants of MARK model, for ablation experiment
'''
we divide the final model into four component:
1.WS(weak supervsion),
2.MTL(multi-task learning),
3.ALW(auto loss weighting),
4.SGN(subtask gated mechanism)
so we gradually turn the baseline(SS:strong supervision) model into our model
'''

class AblationMark_baseline(nn.Module):
    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(AblationMark_baseline, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )
        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length
        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_reg = self.reg(x)
        return output_reg


class AblationMark_classification(nn.Module):
    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(AblationMark_classification, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )
        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length
        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_reg = self.reg(x)
        return output_reg


class AblationMark_Ws(nn.Module):
    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(AblationMark_Ws, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )
        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length
        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_reg = self.reg(x)
        return output_reg


class AblationMark_Ws_Mtl(nn.Module):
    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(AblationMark_Ws_Mtl, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )
        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length
        self.clas = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )
        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_clas = self.clas(x)
        output_reg = self.reg(x)
        return output_clas, output_reg


class AblationMark_Ws_Mtl_Alw(nn.Module):
    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(AblationMark_Ws_Mtl_Alw, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )
        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length
        self.clas = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )
        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_clas = self.clas(x)
        output_reg = self.reg(x)
        return output_clas, output_reg


class AblationMark_Ws_Mtl_Alw_Sgns(nn.Module):
    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(AblationMark_Ws_Mtl_Alw_Sgns, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )
        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length
        self.clas = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )
        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_clas = self.clas(x)
        output_reg = self.reg(x) * self.act(output_clas)
        return output_clas, output_reg


class AblationMark_Ws_Mtl_Alw_Sgnh(nn.Module):
    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(AblationMark_Ws_Mtl_Alw_Sgnh, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim
        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=128, out_channels=524, kernel_size=7, stride=3),
            nn.ReLU(True)
        )
        output_channel, output_length = util_md.compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length
        self.clas = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )
        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )
        self.act = nn.Sigmoid()
        self.threshold = 0.5

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_clas = self.clas(x)
        hard_clas =  (self.act(output_clas)>= self.threshold).float()
        output_reg = self.reg(x) * hard_clas
        return output_clas, output_reg

