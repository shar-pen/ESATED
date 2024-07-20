import torch
import time
import numpy as np
from torch import nn


def compute_output_shape(model:nn.Module, input_shape:list):
    # compute the shape of one net
    input_tensor = torch.zeros(input_shape).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_shape = list(output_tensor.shape[1:])
    return output_shape


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

        output_channel, output_length = compute_output_shape(self.cnn,[1,in_seq_dim])
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


class Model_Deeper(nn.Module):

    def __init__(self, in_seq_dim, out_seq_dim=1) -> None:
        super(Model_Deeper, self).__init__()
        self.in_seq_length = in_seq_dim
        self.out_seq_dim = out_seq_dim

        self.cnn = nn.Sequential(
            nn.ReplicationPad1d((1,1)),
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=30, out_channels=90, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.ReplicationPad1d((2,2)),
            nn.Conv1d(in_channels=90, out_channels=180, kernel_size=5, stride=3),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,3)),
            nn.Conv1d(in_channels=180, out_channels=360, kernel_size=6, stride=3),
            nn.ReLU(True),
            nn.ReplicationPad1d((3,5)),
            nn.Conv1d(in_channels=360, out_channels=720, kernel_size=7, stride=3),
            nn.ReLU(True)
        )

        output_channel, output_length = compute_output_shape(self.cnn,[1,in_seq_dim])
        self.flatten_size = output_channel*output_length

        self.clas = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_seq_dim)
        )

        self.reg = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_seq_dim)
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1,self.flatten_size)
        output_clas = self.clas(x)
        output_reg = self.reg(x) * self.act(output_clas)
        return output_clas, output_reg


model = Model_final(599,14)

data_i = torch.ones((1,1,599))

excute_times = 30
remove_head_cnt = 5

list_t = []

for i in range(excute_times):

    st = time.time()

    data_o_1, data_o_2 = model(data_i)

    ed = time.time()

    list_t.append(ed-st)

print('Time consumption of original model: {:.3f}s'.format(np.mean(list_t[remove_head_cnt:])))


model = Model_Deeper(599,14)

list_t = []

for i in range(excute_times):

    st = time.time()

    data_o_1, data_o_2 = model(data_i)

    ed = time.time()

    list_t.append(ed-st)

print('Time consumption of original model: {:.3f}s'.format(np.mean(list_t[remove_head_cnt:])))
