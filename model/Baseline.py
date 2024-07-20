import random
import numpy as np
import torch
import time
import copy
import math
from torch import nn


class WindowGRU(nn.Module):
    def __init__(self, input_window):
        super(WindowGRU, self).__init__()

        self.input_window = input_window
        self.mid = int(input_window/2)
        # 1D Conv
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1, stride=1)

        # Bi-directional GRUs
        self.gru1 = nn.GRU(input_size=16, hidden_size=64, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 1D Conv
        x = self.conv1d(x)
        x = x.transpose(-2, -1)

        # Bi-directional GRUs
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        x = x[:, self.mid, :]

        # Fully Connected Layers
        x = torch.relu(self.fc1(x))  # Taking the last output of the last GRU layer
        x = self.dropout3(x)
        x = self.fc2(x)

        return x




class DAE(nn.Module):
    def __init__(self, sequence_length):
        # Refer to "KELLY J, KNOTTENBELT W. Neural NILM: Deep neural networks applied to energy disaggregation[C].The 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments".
        super(DAE, self).__init__()
        self.sequence_length = sequence_length
        self.conv_1 = nn.Conv1d(1, 8, 4, stride = 1)
        self.dense = nn.Sequential(nn.Linear(8 * (sequence_length - 3), 8 * (sequence_length - 3)),
                                    nn.ReLU(True), 
                                    nn.Linear(8 * (sequence_length - 3), 128), 
                                    nn.ReLU(True), 
                                    nn.Linear(128, 8 * (sequence_length - 3)), 
                                    nn.ReLU(True))
        self.deconv_2 = nn.ConvTranspose1d(8, 1, 4, stride = 1)

    def forward(self,power_seq):
        inp = self.conv_1(power_seq).view(power_seq.size(0), -1)
        tmp = self.dense(inp).view(power_seq.size(0), 8, -1)
        out = self.deconv_2(tmp)
        return out
    


class BiLSTM(nn.Module):

    def __init__(self, input_window):
        super(BiLSTM, self).__init__()

        self.input_window = input_window
        self.mid = int(input_window/2)
        # 1D Conv
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1, stride=1)

        # Bi-directional GRUs
        self.gru1 = nn.GRU(input_size=16, hidden_size=64, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 1D Conv
        x = self.conv1d(x)
        x = x.transpose(-2, -1)

        # Bi-directional GRUs
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, self.mid, :]

        # Fully Connected Layers
        x = torch.relu(self.fc1(x))  # Taking the last output of the last GRU layer
        x = self.fc2(x)

        return x


class Seq2Point(nn.Module):

    def __init__(self, in_seq_dim, out_seq_dim=1):
        super(Seq2Point, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim

        self.conv = nn.Sequential(
            nn.ReplicationPad1d((4,5)),
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,4)),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,3)),
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d(2),
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d(2),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(True)
        )

        self.dense = nn.Sequential(
            nn.Linear(50 * in_seq_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_seq_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x.view(-1,50 * self.in_seq_length))
        return x.view(-1, self.out_seq_dim)


class Seq2Seq(nn.Module):

    def __init__(self, in_seq_dim, out_seq_dim=1):
        super(Seq2Seq, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim

        self.conv = nn.Sequential(
            nn.ReplicationPad1d((4,5)),
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,4)),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,3)),
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d(2),
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d(2),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(True)
        )

        self.dense = nn.Sequential(
            nn.Linear(50 * in_seq_dim, out_seq_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x.view(-1,50 * self.in_seq_length))
        return x.view(-1, self.out_seq_dim)


class SGN(nn.Module):

    def __init__(self, in_seq_dim):
        super(SGN, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = 1

        self.gate = Seq2Point(in_seq_dim, 1)
        self.reg = Seq2Point(in_seq_dim, 1)
        self.act = nn.Sigmoid()
        self.b = nn.parameter.Parameter(torch.zeros(1))

    def forward(self, x):
        reg_power = self.reg(x)
        app_state = self.gate(x)
        app_state_rsult = self.act(app_state)
        app_power = reg_power * app_state_rsult + (1 - app_state_rsult) * self.b
        return app_state, app_power