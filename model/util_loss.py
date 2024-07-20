import torch
import torch.nn as nn


class LogitAccuracy(nn.Module):

    def __init__(self, reduction='mean'):
        super(LogitAccuracy, self).__init__()
        self.reduction = reduction

    def forward(self, logit, target):
        pred = (torch.sigmoid(logit)>0.5).int()
        acc = (pred == target).float()
        if self.reduction=='none':
            return acc
        elif self.reduction=='mean':
            return acc.mean()
        elif self.reduction=='sum':
            return acc.sum()


class SignalAggregateError(nn.Module):

    def __init__(self, single_class=True, period_len:int=1):
        super(SignalAggregateError, self).__init__()
        self.single_class = single_class
        self.period_len = period_len

    def forward(self, pred, true):
        if self.single_class:
            num_class = 1
            pred = pred.reshape(-1,1)
            true = true.reshape(-1,1)
        else:
            num_class = pred.shape[-1]
            pred = pred.reshape(-1,num_class)
            true = true.reshape(-1,num_class)
        error = pred - true
        num_groups = error.shape[0] // self.period_len
        grouped_tensor = error[:num_groups * self.period_len].reshape(num_groups, self.period_len, num_class)
        grouped_means = torch.mean(grouped_tensor, dim=1)
        result = torch.mean(torch.abs(grouped_means), dim=0)
        return result


class MAE_off_on(nn.Module):

    def __init__(self, single_class=True):
        super(MAE_off_on, self).__init__()
        self.single_class = single_class
        self.metric = nn.L1Loss(reduction='none')

    def forward(self, pred, target, state):
        if self.single_class:
            pred = torch.flatten(pred)
            target = torch.flatten(target)
            state = torch.flatten(state)
        mae = self.metric(pred, target)
        mae_on_mean = (mae*state).sum(dim=0)/state.sum(dim=0)
        mae_off_mean = (mae*(1-state)).sum(dim=0)/(1-state).sum(dim=0)
        mae_01_mean = torch.vstack([mae_off_mean, mae_on_mean])
        mae_01_mean = torch.where(torch.isnan(mae_01_mean), torch.tensor(0.0, device=pred.device), mae_01_mean)
        return mae_01_mean


class MAE(nn.Module):

    def __init__(self, single_class=True):
        super(MAE, self).__init__()
        self.single_class = single_class
        self.metric = nn.L1Loss(reduction='none')

    def forward(self, pred, target):
        if self.single_class:
            pred = torch.flatten(pred)
            target = torch.flatten(target)
        mae = self.metric(pred, target)
        if self.single_class:
            mae = mae.mean()
        else:
            mae = mae.mean(dim=0)
        return mae



class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum