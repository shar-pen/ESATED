import os
import torch
import numpy as np
import pandas as pd
from torch import nn
import time
import copy
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import model.util_loss as util_ls

TAB_str = '    '


def initialize(layer):
    # Xavier_uniform will be applied to conv1d and dense layer, to be sonsistent with Keras and Tensorflow
    if isinstance(layer,nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val = 0.0)


class ModelWeightManager():

    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def save_model_weight(self, model:nn.Module, pth_name):
        # only save model params
        save_directory = os.path.join(self.file_path, pth_name + '.pth')
        torch.save(model.state_dict(), save_directory)
        print("Please find file in: ", save_directory)

    def load_model_weight(self, model:nn.Module, pth_name):
        save_directory = os.path.join(self.file_path, pth_name + '.pth')
        print("Load model file from: ", save_directory)
        model.load_state_dict(torch.load(save_directory))

    def save_model_whole(self, model:nn.Module, pth_name):
        # save all model
        save_directory = os.path.join(self.file_path, pth_name + '.pth')
        torch.save(model, save_directory)
        print("Please find file in: ", save_directory)

    def load_model_whole(self, pth_name):
        save_directory = os.path.join(self.file_path, pth_name + '.pth')
        print("Load model file from: ", save_directory)
        model = torch.load(save_directory)
        return model


class DictManager():

    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def save_dict(self, data, pth_name):
        save_directory = os.path.join(self.file_path, pth_name)
        torch.save(data, save_directory)
        print("Please find file in: ", save_directory)

    def load_dict(self, pth_name):
        save_directory = os.path.join(self.file_path, pth_name)
        print("Load model file from: ", save_directory)
        data = torch.load(save_directory)
        return data


def print_model_structure(model):
    for name, param in model.named_parameters():
        print(f'{name} [{param.requires_grad}]')


def train_val_classification(device:torch.device, model:nn.Module, epochs:int=1,
                trainset:DataLoader=None, valset:DataLoader=None,
                loss_fn=nn.MSELoss(), loss_factor=1,
                return_metric_dict={'MAE(apps)':util_ls.MAE(single_class=False)},
                lr:float=1e-3, decay_params=None):
    # train and eval function for normal model
    # dataset

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    # save data
    metric_score_list = {phase:{metric_name:[] for metric_name in return_metric_dict} for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            metric_score_epoch = {metric_name:0.0 for metric_name in return_metric_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):

                for i, (batch_input, batch_output) in enumerate(dataloaders[phase]):

                    # get the inputs
                    batch_input = batch_input.to(device)
                    batch_output = batch_output.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # predict
                    batch_pred = model(batch_input)
                    # compute the loss based on model output and real labels
                    batch_loss = loss_fn(batch_pred, batch_output)

                    epoch_loss += batch_loss.item()
                    for metric_name, metric_fn in return_metric_dict.items():
                        tmp_loss = metric_fn(batch_pred, batch_output).mean(dim=0)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        # backpropagate the loss
                        batch_loss.backward()
                        # adjust parameters based on the calculated gradients
                        optimizer.step()

            epoch_loss_factor = loss_factor/batch_num

            epoch_loss *= epoch_loss_factor
            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            for metric_name, value in metric_score_epoch.items():
                value *= epoch_loss_factor
                metric_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')


            # Save best only
            compare_var = epoch_loss
            if train_flag and phase == compare_best_phase and \
                (best_loss is None or compare_var < best_loss):
                best_loss = compare_var
                best_loss_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")

        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_score_list.items():
        for metric_name, data in phase_score.items():
            metric_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_score_list['Best epoch'] = best_loss_epoch

    return metric_score_list


def train_val(device:torch.device, model:nn.Module, epochs:int=1,
                trainset:DataLoader=None, valset:DataLoader=None,
                loss_fn=nn.MSELoss(), loss_factor=1,
                return_metric_dict={'MAE(apps)':util_ls.MAE(single_class=True),
                                    'SAE':util_ls.SignalAggregateError(single_class=True, period_len=450),
                                    'MAE(offon)':util_ls.MAE_off_on(single_class=True)},
                lr:float=1e-3, decay_params=None):
    # train and eval function for normal model
    # dataset

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    # save data
    metric_score_list = {phase:{metric_name:[] for metric_name in return_metric_dict} for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            metric_score_epoch = {metric_name:0.0 for metric_name in return_metric_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):

                for i, (batch_input, batch_output, batch_state) in enumerate(dataloaders[phase]):

                    # get the inputs
                    batch_input = batch_input.to(device)
                    batch_output = batch_output.to(device)
                    batch_state = batch_state.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # predict
                    batch_pred = model(batch_input)
                    # compute the loss based on model output and real labels
                    batch_loss = loss_fn(batch_pred, batch_output)

                    epoch_loss += batch_loss.item()
                    for metric_name, metric_fn in return_metric_dict.items():
                        if isinstance(metric_fn, util_ls.MAE_off_on):
                            tmp_loss = metric_fn(batch_pred, batch_output, batch_state)
                        else:
                            tmp_loss = metric_fn(batch_pred, batch_output)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        # backpropagate the loss
                        batch_loss.backward()
                        # adjust parameters based on the calculated gradients
                        optimizer.step()

            epoch_loss_factor = loss_factor/batch_num

            epoch_loss *= epoch_loss_factor
            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            for metric_name, value in metric_score_epoch.items():
                value *= epoch_loss_factor
                metric_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')


            # Save best only
            compare_var = epoch_loss
            if train_flag and phase == compare_best_phase and \
                (best_loss is None or compare_var < best_loss):
                best_loss = compare_var
                best_loss_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")

        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_score_list.items():
        for metric_name, data in phase_score.items():
            metric_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_score_list['Best epoch'] = best_loss_epoch

    return metric_score_list


def predict(device:torch.device, model:nn.Module, input:DataLoader=None)->np.array:
    # predict for input, return numpy array

    print("The model will be running on", device, "device")
    model.to(device)

    first_flag = True
    multiple_output_flag = False
    model.eval()
    with torch.no_grad():
        for i, (batch_input, ) in enumerate(input):
            batch_input = batch_input.to(device)
            batch_pred = model(batch_input)
            if first_flag:
                first_flag = False
                if isinstance(batch_pred, tuple):
                    multiple_output_flag = True
                if multiple_output_flag:
                    pred_list = batch_pred
                else:
                    pred_list = [batch_pred]
            else:
                if not multiple_output_flag:
                    batch_pred = [batch_pred]
                pred_list = [torch.cat((a, b), dim=0) for _, (a,b) in enumerate(zip(pred_list, batch_pred))]

    pred_list = [item.cpu().numpy() for item in pred_list]
    return pred_list


def compute_output_shape(model:nn.Module, input_shape:list):
    # compute the shape of one net
    input_tensor = torch.zeros(input_shape).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_shape = list(output_tensor.shape[1:])
    return output_shape


def train_val_Mark(device:torch.device, model:nn.Module, epochs:int=1,
                    trainset:DataLoader=None, valset:DataLoader=None,
                    loss_reg=nn.MSELoss(), loss_factor=1,
                    return_metric_dict={'MAE(apps)':util_ls.MAE(single_class=True),
                                        'SAE':util_ls.SignalAggregateError(single_class=True, period_len=450),
                                        'MAE(offon)':util_ls.MAE_off_on(single_class=True)},
                    lr:float=1e-3, decay_params=None):

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    # save data
    metric_score_list = {phase:{metric_name:[] for metric_name in return_metric_dict} for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            epoch_loss_sum = 0.0
            epoch_loss_zero = 0.0
            epoch_loss_real = 0.0
            metric_score_epoch = {metric_name:0.0 for metric_name in return_metric_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):
                for i, (batch_power_main, batch_state_apps, batch_power_main_midpoint, batch_power_apps) in enumerate(dataloaders[phase]):

                    batch_power_main = batch_power_main.to(device)
                    batch_state_apps = batch_state_apps.to(device)
                    batch_power_main_midpoint = batch_power_main_midpoint.to(device)
                    batch_power_apps = batch_power_apps.to(device)

                    optimizer.zero_grad()

                    batch_pred = model(batch_power_main)
                    batch_pred_sum = (batch_pred*batch_state_apps).sum(dim=-1, keepdim=True)
                    batch_off_pred = batch_pred*(1-batch_state_apps)

                    loss_sum = loss_reg(batch_pred_sum, batch_power_main_midpoint)
                    loss_zero = loss_reg(batch_off_pred, torch.zeros_like(batch_off_pred))
                    batch_loss = loss_sum + loss_zero

                    epoch_loss += batch_loss.item()
                    epoch_loss_sum += loss_sum.item()
                    epoch_loss_zero += loss_zero.item()
                    epoch_loss_real += loss_reg(batch_pred, batch_power_apps).item()

                    for metric_name, metric_fn in return_metric_dict.items():
                        if isinstance(metric_fn, util_ls.MAE_off_on):
                            tmp_loss = metric_fn(batch_pred, batch_power_apps, batch_state_apps)
                        else:
                            tmp_loss = metric_fn(batch_pred, batch_power_apps)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

            epoch_loss_factor = loss_factor/batch_num

            epoch_loss *= epoch_loss_factor
            epoch_loss_sum *= epoch_loss_factor
            epoch_loss_zero *= epoch_loss_factor
            epoch_loss_real *= epoch_loss_factor

            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            print(TAB_str, f'epoch_loss_sum: {epoch_loss_sum}')
            print(TAB_str, f'epoch_loss_zero: {epoch_loss_zero}')
            print(TAB_str, f'epoch_loss_real: {epoch_loss_real}')

            for metric_name, value in metric_score_epoch.items():
                value *= epoch_loss_factor
                metric_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            compare_var = epoch_loss_real
            if train_flag and phase == compare_best_phase and \
                    (best_loss is None or compare_var < best_loss):
                    best_loss = compare_var
                    best_loss_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")

        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_score_list.items():
        for metric_name, data in phase_score.items():
            metric_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_score_list['Best epoch'] = best_loss_epoch

    return metric_score_list


def train_val_Mark_DoubleTask(device:torch.device, model:nn.Module, epochs:int=1, trainset:DataLoader=None, valset:DataLoader=None,
                    loss_reg=nn.MSELoss(), loss_clas=nn.BCEWithLogitsLoss(),
                    return_metric_reg_dict={'MAE(apps)':util_ls.MAE(single_class=True),
                                            'SAE':util_ls.SignalAggregateError(single_class=True, period_len=450),
                                            'MAE(offon)':util_ls.MAE_off_on(single_class=True)},
                    return_metric_clas_dict={'Acc(apps)':util_ls.LogitAccuracy(reduction='none'), 'Acc(mean)':util_ls.LogitAccuracy()},
                    loss_reg_factor=1, lr:float=1e-3, decay_params=None):

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)


    # save data
    metric_reg_score_list = {phase:{metric_name:[] for metric_name in return_metric_reg_dict}
                             for phase in data_phase}
    metric_clas_score_list = {phase:{metric_name:[] for metric_name in return_metric_clas_dict}
                              for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            epoch_loss_sum = 0.0
            epoch_loss_zero = 0.0
            epoch_loss_real = 0.0
            epoch_loss_clas = 0.0
            metric_reg_score_epoch = {metric_name:0.0 for metric_name in return_metric_reg_dict}
            metric_clas_score_epoch = {metric_name:0.0 for metric_name in return_metric_clas_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):
                for i, (batch_power_main, batch_state_apps, batch_power_main_midpoint, batch_power_apps) in enumerate(dataloaders[phase]):

                    batch_power_main = batch_power_main.to(device)
                    batch_state_apps = batch_state_apps.to(device)
                    batch_power_main_midpoint = batch_power_main_midpoint.to(device)
                    batch_power_apps = batch_power_apps.to(device)

                    optimizer.zero_grad()

                    batch_pred_state, batch_pred_power = model(batch_power_main)
                    batch_pred_sum = (batch_pred_power*batch_state_apps).sum(dim=-1, keepdim=True)
                    batch_off_pred = batch_pred_power*(1-batch_state_apps)

                    loss_sum = loss_reg(batch_pred_sum, batch_power_main_midpoint)
                    loss_zero = loss_reg(batch_off_pred, torch.zeros_like(batch_off_pred))
                    loss_state = loss_clas(batch_pred_state, batch_state_apps)
                    batch_loss_reg = loss_sum + loss_zero
                    batch_loss_clas = loss_state
                    batch_loss = batch_loss_clas + batch_loss_reg

                    epoch_loss += batch_loss.item()
                    epoch_loss_sum += loss_sum.item()
                    epoch_loss_zero += loss_zero.item()
                    epoch_loss_real += loss_reg(batch_pred_power, batch_power_apps).item()
                    epoch_loss_clas += batch_loss_clas.item()


                    for metric_name, metric_fn in return_metric_reg_dict.items():
                        if isinstance(metric_fn, util_ls.MAE_off_on):
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps, batch_state_apps)
                        else:
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_reg_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    for metric_name, metric_fn in return_metric_clas_dict.items():
                        tmp_loss = metric_fn(batch_pred_state, batch_state_apps).mean(dim=0)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_clas_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

            epoch_loss_factor = loss_reg_factor/batch_num

            epoch_loss *= epoch_loss_factor
            epoch_loss_sum *= epoch_loss_factor
            epoch_loss_zero *= epoch_loss_factor
            epoch_loss_real *= epoch_loss_factor
            epoch_loss_clas /= batch_num

            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            print(TAB_str, f'epoch_loss_sum: {epoch_loss_sum}')
            print(TAB_str, f'epoch_loss_zero: {epoch_loss_zero}')
            print(TAB_str, f'epoch_loss_real: {epoch_loss_real}')
            print(TAB_str, f'epoch_loss_clas: {epoch_loss_clas}')

            for metric_name, value in metric_reg_score_epoch.items():
                value *= epoch_loss_factor
                metric_reg_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            for metric_name, value in metric_clas_score_epoch.items():
                value /= batch_num
                metric_clas_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            compare_var = epoch_loss_real
            if train_flag and phase == compare_best_phase and \
                    (best_loss is None or compare_var < best_loss):
                    best_loss = compare_var
                    best_loss_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")

        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_reg_score_list.items():
        for metric_name, data in phase_score.items():
            metric_reg_score_list[phase][metric_name] = np.array(data)

    for phase, phase_score in metric_clas_score_list.items():
        for metric_name, data in phase_score.items():
            metric_clas_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_reg_score_list['Best epoch'] = best_loss_epoch
        metric_clas_score_list['Best epoch'] = best_loss_epoch

    return metric_reg_score_list, metric_clas_score_list


def train_val_Mark_DoubleTask_AutomaticWeightedLoss(device:torch.device, model:nn.Module, epochs:int=1,
                    trainset:DataLoader=None, valset:DataLoader=None,
                    loss_reg=nn.MSELoss(), loss_clas=nn.BCEWithLogitsLoss(),
                    return_metric_reg_dict={'MAE(apps)':util_ls.MAE(single_class=True),
                                            'SAE':util_ls.SignalAggregateError(single_class=True, period_len=450),
                                            'MAE(offon)':util_ls.MAE_off_on(single_class=True)},
                    return_metric_clas_dict={'Acc(apps)':util_ls.LogitAccuracy(reduction='none'), 'Acc(mean)':util_ls.LogitAccuracy()},
                    loss_reg_factor=1, lr:float=1e-3, decay_params=None):

    '''
    return_metric_reg_dict={'MAE(apps)':nn.L1Loss(reduction='none'), 'MAE(mean)':nn.L1Loss(), 'SAE':util_ls.SignalAggregateError(reduction='none')},
    return_metric_clas_dict={'Acc(apps)':util_ls.LogitAccuracy(reduction='none'), 'Acc(mean)':util_ls.LogitAccuracy()},
    '''

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    awl_reg = util_ls.AutomaticWeightedLoss(2)
    awl_task = util_ls.AutomaticWeightedLoss(2)
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': awl_reg.parameters()},
                {'params': awl_task.parameters()}
            ], lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    # save data
    metric_reg_score_list = {phase:{metric_name:[] for metric_name in return_metric_reg_dict}
                             for phase in data_phase}
    metric_clas_score_list = {phase:{metric_name:[] for metric_name in return_metric_clas_dict}
                              for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            epoch_loss_sum = 0.0
            epoch_loss_zero = 0.0
            epoch_loss_real = 0.0
            epoch_loss_clas = 0.0
            metric_reg_score_epoch = {metric_name:0.0 for metric_name in return_metric_reg_dict}
            metric_clas_score_epoch = {metric_name:0.0 for metric_name in return_metric_clas_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):
                for i, (batch_power_main, batch_state_apps, batch_power_main_midpoint, batch_power_apps) in enumerate(dataloaders[phase]):

                    batch_power_main = batch_power_main.to(device)
                    batch_state_apps = batch_state_apps.to(device)
                    batch_power_main_midpoint = batch_power_main_midpoint.to(device)
                    batch_power_apps = batch_power_apps.to(device)

                    optimizer.zero_grad()

                    batch_pred_state, batch_pred_power = model(batch_power_main)
                    batch_pred_sum = (batch_pred_power*batch_state_apps).sum(dim=-1, keepdim=True)
                    batch_off_pred = batch_pred_power*(1-batch_state_apps)

                    loss_sum = loss_reg(batch_pred_sum, batch_power_main_midpoint)
                    loss_zero = loss_reg(batch_off_pred, torch.zeros_like(batch_off_pred))
                    loss_state = loss_clas(batch_pred_state, batch_state_apps)

                    batch_loss = awl_task(loss_state, awl_reg(loss_sum, loss_zero))
                    # awl(loss_sum, loss_zero, loss_state)

                    epoch_loss += batch_loss.item()
                    epoch_loss_sum += loss_sum.item()
                    epoch_loss_zero += loss_zero.item()
                    epoch_loss_real += loss_reg(batch_pred_power, batch_power_apps).item()
                    epoch_loss_clas += loss_state.item()


                    for metric_name, metric_fn in return_metric_reg_dict.items():
                        if isinstance(metric_fn, util_ls.MAE_off_on):
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps, batch_state_apps)
                        else:
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_reg_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    for metric_name, metric_fn in return_metric_clas_dict.items():
                        tmp_loss = metric_fn(batch_pred_state, batch_state_apps).mean(dim=0)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_clas_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

            epoch_loss_factor = loss_reg_factor/batch_num

            epoch_loss *= epoch_loss_factor
            epoch_loss_sum *= epoch_loss_factor
            epoch_loss_zero *= epoch_loss_factor
            epoch_loss_real *= epoch_loss_factor
            epoch_loss_clas /= batch_num

            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            print(TAB_str, f'epoch_loss_sum: {epoch_loss_sum}')
            print(TAB_str, f'epoch_loss_zero: {epoch_loss_zero}')
            print(TAB_str, f'epoch_loss_real: {epoch_loss_real}')
            print(TAB_str, f'epoch_loss_clas: {epoch_loss_clas}')

            for metric_name, value in metric_reg_score_epoch.items():
                value *= epoch_loss_factor
                metric_reg_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            for metric_name, value in metric_clas_score_epoch.items():
                value /= batch_num
                metric_clas_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            compare_var = epoch_loss_real
            if train_flag and phase == compare_best_phase and \
                    (best_loss is None or compare_var < best_loss):
                    best_loss = compare_var
                    best_loss_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")

        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

        for name, parameter in awl_task.named_parameters():
            print(f"Task Parameter values: {parameter}")
        for name, parameter in awl_reg.named_parameters():
            print(f"Reg Parameter values: {parameter}")

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_reg_score_list.items():
        for metric_name, data in phase_score.items():
            metric_reg_score_list[phase][metric_name] = np.array(data)

    for phase, phase_score in metric_clas_score_list.items():
        for metric_name, data in phase_score.items():
            metric_clas_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_reg_score_list['Best epoch'] = best_loss_epoch
        metric_clas_score_list['Best epoch'] = best_loss_epoch

    return metric_reg_score_list, metric_clas_score_list



def train_val_Mark_DoubleTask_AutomaticWeightedLoss_1L(device:torch.device, model:nn.Module, epochs:int=1,
                    trainset:DataLoader=None, valset:DataLoader=None,
                    loss_reg=nn.MSELoss(), loss_clas=nn.BCEWithLogitsLoss(),
                    return_metric_reg_dict={'MAE(apps)':util_ls.MAE(single_class=True),
                                            'SAE':util_ls.SignalAggregateError(single_class=True, period_len=450),
                                            'MAE(offon)':util_ls.MAE_off_on(single_class=True)},
                    return_metric_clas_dict={'Acc(apps)':util_ls.LogitAccuracy(reduction='none'), 'Acc(mean)':util_ls.LogitAccuracy()},
                    loss_reg_factor=1, lr:float=1e-3, decay_params=None):

    '''
    return_metric_reg_dict={'MAE(apps)':nn.L1Loss(reduction='none'), 'MAE(mean)':nn.L1Loss(), 'SAE':util_ls.SignalAggregateError(reduction='none')},
    return_metric_clas_dict={'Acc(apps)':util_ls.LogitAccuracy(reduction='none'), 'Acc(mean)':util_ls.LogitAccuracy()},
    '''

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    awl = util_ls.AutomaticWeightedLoss(3)
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': awl.parameters()}
            ], lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    # save data
    metric_reg_score_list = {phase:{metric_name:[] for metric_name in return_metric_reg_dict}
                             for phase in data_phase}
    metric_clas_score_list = {phase:{metric_name:[] for metric_name in return_metric_clas_dict}
                              for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            epoch_loss_sum = 0.0
            epoch_loss_zero = 0.0
            epoch_loss_real = 0.0
            epoch_loss_clas = 0.0
            metric_reg_score_epoch = {metric_name:0.0 for metric_name in return_metric_reg_dict}
            metric_clas_score_epoch = {metric_name:0.0 for metric_name in return_metric_clas_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):
                for i, (batch_power_main, batch_state_apps, batch_power_main_midpoint, batch_power_apps) in enumerate(dataloaders[phase]):

                    batch_power_main = batch_power_main.to(device)
                    batch_state_apps = batch_state_apps.to(device)
                    batch_power_main_midpoint = batch_power_main_midpoint.to(device)
                    batch_power_apps = batch_power_apps.to(device)

                    optimizer.zero_grad()

                    batch_pred_state, batch_pred_power = model(batch_power_main)
                    batch_pred_sum = (batch_pred_power*batch_state_apps).sum(dim=-1, keepdim=True)
                    batch_off_pred = batch_pred_power*(1-batch_state_apps)

                    loss_sum = loss_reg(batch_pred_sum, batch_power_main_midpoint)
                    loss_zero = loss_reg(batch_off_pred, torch.zeros_like(batch_off_pred))
                    loss_state = loss_clas(batch_pred_state, batch_state_apps)

                    batch_loss = awl(loss_state, loss_sum, loss_zero)
                    # awl(loss_sum, loss_zero, loss_state)

                    epoch_loss += batch_loss.item()
                    epoch_loss_sum += loss_sum.item()
                    epoch_loss_zero += loss_zero.item()
                    epoch_loss_real += loss_reg(batch_pred_power, batch_power_apps).item()
                    epoch_loss_clas += loss_state.item()


                    for metric_name, metric_fn in return_metric_reg_dict.items():
                        if isinstance(metric_fn, util_ls.MAE_off_on):
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps, batch_state_apps)
                        else:
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_reg_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    for metric_name, metric_fn in return_metric_clas_dict.items():
                        tmp_loss = metric_fn(batch_pred_state, batch_state_apps).mean(dim=0)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_clas_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

            epoch_loss_factor = loss_reg_factor/batch_num

            epoch_loss *= epoch_loss_factor
            epoch_loss_sum *= epoch_loss_factor
            epoch_loss_zero *= epoch_loss_factor
            epoch_loss_real *= epoch_loss_factor
            epoch_loss_clas /= batch_num

            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            print(TAB_str, f'epoch_loss_sum: {epoch_loss_sum}')
            print(TAB_str, f'epoch_loss_zero: {epoch_loss_zero}')
            print(TAB_str, f'epoch_loss_real: {epoch_loss_real}')
            print(TAB_str, f'epoch_loss_clas: {epoch_loss_clas}')

            for metric_name, value in metric_reg_score_epoch.items():
                value *= epoch_loss_factor
                metric_reg_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            for metric_name, value in metric_clas_score_epoch.items():
                value /= batch_num
                metric_clas_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            compare_var = epoch_loss_real
            if train_flag and phase == compare_best_phase and \
                    (best_loss is None or compare_var < best_loss):
                    best_loss = compare_var
                    best_loss_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")

        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

        for name, parameter in awl.named_parameters():
            print(f"ALW Parameter values: {parameter}")

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_reg_score_list.items():
        for metric_name, data in phase_score.items():
            metric_reg_score_list[phase][metric_name] = np.array(data)

    for phase, phase_score in metric_clas_score_list.items():
        for metric_name, data in phase_score.items():
            metric_clas_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_reg_score_list['Best epoch'] = best_loss_epoch
        metric_clas_score_list['Best epoch'] = best_loss_epoch

    return metric_reg_score_list, metric_clas_score_list



def train_val_DoubleTask(device:torch.device, model:nn.Module, epochs:int=1, trainset:DataLoader=None, valset:DataLoader=None,
                    loss_reg=nn.MSELoss(), loss_clas=nn.BCEWithLogitsLoss(),
                    return_metric_reg_dict={'MAE(apps)':util_ls.MAE(single_class=True),
                                            'SAE':util_ls.SignalAggregateError(single_class=True, period_len=450),
                                            'MAE(offon)':util_ls.MAE_off_on(single_class=True)},
                    return_metric_clas_dict={'Acc(apps)':util_ls.LogitAccuracy(reduction='none'), 'Acc(mean)':util_ls.LogitAccuracy()},
                    loss_reg_factor=1, lr:float=1e-3, decay_params=None):

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    # save data
    metric_reg_score_list = {phase:{metric_name:[] for metric_name in return_metric_reg_dict}
                             for phase in data_phase}
    metric_clas_score_list = {phase:{metric_name:[] for metric_name in return_metric_clas_dict}
                              for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            epoch_loss_reg = 0.0
            epoch_loss_clas = 0.0
            metric_reg_score_epoch = {metric_name:0.0 for metric_name in return_metric_reg_dict}
            metric_clas_score_epoch = {metric_name:0.0 for metric_name in return_metric_clas_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):
                for i, (batch_power_main, batch_power_apps, batch_state_apps) in enumerate(dataloaders[phase]):

                    batch_power_main = batch_power_main.to(device)
                    batch_power_apps = batch_power_apps.to(device)
                    batch_state_apps = batch_state_apps.to(device)

                    optimizer.zero_grad()

                    batch_pred_state, batch_pred_power = model(batch_power_main)

                    batch_loss_clas = loss_clas(batch_pred_state, batch_state_apps)
                    batch_loss_reg = loss_reg(batch_pred_power, batch_power_apps)
                    batch_loss = batch_loss_clas + batch_loss_reg

                    epoch_loss += batch_loss.item()
                    epoch_loss_clas += batch_loss_clas.item()
                    epoch_loss_reg += batch_loss_reg.item()


                    for metric_name, metric_fn in return_metric_reg_dict.items():
                        if isinstance(metric_fn, util_ls.MAE_off_on):
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps, batch_state_apps)
                        else:
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_reg_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    for metric_name, metric_fn in return_metric_clas_dict.items():
                        tmp_loss = metric_fn(batch_pred_state, batch_state_apps).mean(dim=0)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_clas_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

            epoch_loss_factor = loss_reg_factor/batch_num

            epoch_loss *= epoch_loss_factor
            epoch_loss_reg *= epoch_loss_factor
            epoch_loss_clas /= batch_num

            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            print(TAB_str, f'epoch_loss_reg: {epoch_loss_reg}')
            print(TAB_str, f'epoch_loss_clas: {epoch_loss_clas}')

            for metric_name, value in metric_reg_score_epoch.items():
                value *= epoch_loss_factor
                metric_reg_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            for metric_name, value in metric_clas_score_epoch.items():
                value /= batch_num
                metric_clas_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            compare_var = epoch_loss
            if train_flag and phase == compare_best_phase and \
                    (best_loss is None or compare_var < best_loss):
                    best_loss = compare_var
                    best_loss_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")


        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_reg_score_list.items():
        for metric_name, data in phase_score.items():
            metric_reg_score_list[phase][metric_name] = np.array(data)

    for phase, phase_score in metric_clas_score_list.items():
        for metric_name, data in phase_score.items():
            metric_clas_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_reg_score_list['Best epoch'] = best_loss_epoch
        metric_clas_score_list['Best epoch'] = best_loss_epoch

    return metric_reg_score_list, metric_clas_score_list



def train_val_DoubleTask_AutomaticWeightedLoss_1L(device:torch.device, model:nn.Module, epochs:int=1, trainset:DataLoader=None, valset:DataLoader=None,
                    loss_reg=nn.MSELoss(), loss_clas=nn.BCEWithLogitsLoss(),
                    return_metric_reg_dict={'MAE(apps)':util_ls.MAE(single_class=True),
                                            'SAE':util_ls.SignalAggregateError(single_class=True, period_len=450),
                                            'MAE(offon)':util_ls.MAE_off_on(single_class=True)},
                    return_metric_clas_dict={'Acc(apps)':util_ls.LogitAccuracy(reduction='none'), 'Acc(mean)':util_ls.LogitAccuracy()},
                    loss_reg_factor=1, lr:float=1e-3, decay_params=None):

    print("The model will be running on", device, "device")
    model.to(device)

    # dataloaders init
    dataloaders = {}
    data_phase = []
    if trainset:
        dataloaders['train'] = trainset
        data_phase.append('train')
    if valset:
        dataloaders['val'] = valset
        data_phase.append('val')
    assert data_phase != []
    if data_phase == ['val']:
        epochs = 1
        train_flag = False
    else:
        train_flag = True
    if 'val' in data_phase:
        compare_best_phase = 'val'
    elif 'train' in data_phase:
        compare_best_phase = 'train'
    else:
        compare_best_phase = None

    # init opt and decay
    awl = util_ls.AutomaticWeightedLoss(2)
    lr_decay_flag = False
    if not decay_params is None:
        lr_decay_flag = train_flag & True
        decay_step = decay_params[0]
        decay_gamma = decay_params[1]
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': awl.parameters()}
            ], lr=lr)
    if lr_decay_flag:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    # save data
    metric_reg_score_list = {phase:{metric_name:[] for metric_name in return_metric_reg_dict}
                             for phase in data_phase}
    metric_clas_score_list = {phase:{metric_name:[] for metric_name in return_metric_clas_dict}
                              for phase in data_phase}
    best_loss = None

    since = time.time()
    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs-1))

        st = time.time()

        for phase in data_phase:

            batch_num = len(dataloaders[phase])
            epoch_loss = 0.0
            epoch_loss_reg = 0.0
            epoch_loss_clas = 0.0
            metric_reg_score_epoch = {metric_name:0.0 for metric_name in return_metric_reg_dict}
            metric_clas_score_epoch = {metric_name:0.0 for metric_name in return_metric_clas_dict}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            with torch.set_grad_enabled(phase == 'train'):
                for i, (batch_power_main, batch_power_apps, batch_state_apps) in enumerate(dataloaders[phase]):

                    batch_power_main = batch_power_main.to(device)
                    batch_power_apps = batch_power_apps.to(device)
                    batch_state_apps = batch_state_apps.to(device)

                    optimizer.zero_grad()

                    batch_pred_state, batch_pred_power = model(batch_power_main)

                    batch_loss_clas = loss_clas(batch_pred_state, batch_state_apps)
                    batch_loss_reg = loss_reg(batch_pred_power, batch_power_apps)

                    batch_loss = awl(batch_loss_clas, batch_loss_reg)

                    epoch_loss += batch_loss.item()
                    epoch_loss_clas += batch_loss_clas.item()
                    epoch_loss_reg += batch_loss_reg.item()


                    for metric_name, metric_fn in return_metric_reg_dict.items():
                        if isinstance(metric_fn, util_ls.MAE_off_on):
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps, batch_state_apps)
                        else:
                            tmp_loss = metric_fn(batch_pred_power, batch_power_apps)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_reg_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    for metric_name, metric_fn in return_metric_clas_dict.items():
                        tmp_loss = metric_fn(batch_pred_state, batch_state_apps).mean(dim=0)
                        if not torch.any(torch.isnan(tmp_loss)):
                            metric_clas_score_epoch[metric_name] += tmp_loss.detach().cpu().numpy()

                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

            epoch_loss_factor = loss_reg_factor/batch_num

            epoch_loss *= epoch_loss_factor
            epoch_loss_reg *= epoch_loss_factor
            epoch_loss_clas /= batch_num

            print(TAB_str, f'Phase: {phase} | Loss: {epoch_loss}')
            print(TAB_str, f'epoch_loss_reg: {epoch_loss_reg}')
            print(TAB_str, f'epoch_loss_clas: {epoch_loss_clas}')

            for metric_name, value in metric_reg_score_epoch.items():
                value *= epoch_loss_factor
                metric_reg_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            for metric_name, value in metric_clas_score_epoch.items():
                value /= batch_num
                metric_clas_score_list[phase][metric_name].append(value)
                print(TAB_str * 2, f'Metric {metric_name} score: {value}')

            compare_var = epoch_loss
            if train_flag and phase == compare_best_phase and \
                    (best_loss is None or compare_var < best_loss):
                    best_loss = compare_var
                    best_loss_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        if lr_decay_flag:
            scheduler.step()
            print(TAB_str, f"Learning Rate: {scheduler.get_last_lr()[0]}")


        ed = time.time()
        print(TAB_str, 'Time consumption: {:.0f}s'.format(ed-st))

        for name, parameter in awl.named_parameters():
            print(f"ALW Parameter values: {parameter}")

    time_elapsed = time.time() - since
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if train_flag:
        print(f'Best loss: {best_loss} in epoch {best_loss_epoch}')
        model.load_state_dict(best_model_wts)

    for phase, phase_score in metric_reg_score_list.items():
        for metric_name, data in phase_score.items():
            metric_reg_score_list[phase][metric_name] = np.array(data)

    for phase, phase_score in metric_clas_score_list.items():
        for metric_name, data in phase_score.items():
            metric_clas_score_list[phase][metric_name] = np.array(data)

    if train_flag:
        metric_reg_score_list['Best epoch'] = best_loss_epoch
        metric_clas_score_list['Best epoch'] = best_loss_epoch

    return metric_reg_score_list, metric_clas_score_list

