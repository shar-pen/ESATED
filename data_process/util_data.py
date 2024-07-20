import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta


def plot_record_with_metric(data:dict, need_to_average:bool=False):
    phases = list(data.keys())
    phase_num = len(phases)
    if isinstance(data[phases[0]], np.ndarray):
        fig, axs = plt.subplots(phase_num, 1)
        for i in range(phase_num):
            p = phases[i]
            tmp = data[p]
            if need_to_average:
                tmp = np.mean(tmp, axis=1)
            axs[i].plot(tmp)
            axs[i].set_title('Phase {}'.format(p))
    else:
        metrics = list(data[phases[0]].keys())
        metric_num = len(metrics)
        fig, axs = plt.subplots(phase_num, metric_num)
        for i in range(phase_num):
            for j in range(metric_num):
                p = phases[i]
                m = metrics[j]
                tmp = data[p][m]
                if need_to_average:
                    tmp = np.mean(tmp, axis=1)
                axs[i][j].plot(tmp)
                axs[i][j].set_title('Phase {} | Metric {}'.format(p, m))


def get_positive_weight(label:np.array, print_ratio_flag:bool=True):
    # get the num of postive and negative labeled sample
    # calculate the weight ratio
    num_samples, num_labels = label.shape
    pos_num = np.sum(label, axis=0)
    neg_num = num_samples - pos_num
    pos_weight = neg_num/pos_num
    pos_ratio = pos_num/num_samples
    neg_ratio = neg_num/num_samples
    if print_ratio_flag:
        print('Pos ratio:',["L{}:{:.2f}%".format(i, num*100) for i, num in enumerate(pos_ratio)])
        print('Neg ratio:',["L{}:{:.2f}%".format(i, num*100) for i, num in enumerate(neg_ratio)])
    return pos_weight, pos_ratio, neg_ratio


def count_label(label:np.array):
    # count the appear time of each label in a multi-hot label
    unique_codes, counts = np.unique(label, axis=0, return_counts=True)
    print(f'Code type num: {len(unique_codes)}; Code num: {label.shape[0]}')
    for code, count in zip(unique_codes, counts):
        print(f"Code {code} appeared {count}")
    print(f"Code averagly appeared {np.mean(counts)}")
    return unique_codes, counts


def normalization_zero_mean(data, max_zero_flag:bool=True):
    # normalize data using zero-mean norm, can shrink max to 1 if flag=true
    mean = data.mean()
    std = data.std()
    norml_data = (data - mean)/std
    if max_zero_flag:
        max = np.abs(norml_data).max()
        norml_data = norml_data/max
    else:
        max = 1
    return norml_data, mean, std, max


def normalization_interval(data, a, b):
    # normalize data to [a,b]
    # double use it to reverse data
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    norm_data = (data - min_val)/range_val* (b - a) + a
    return norm_data, min_val, max_val


def print_shape(*data_seq):
    # print shape of variables
    for d in data_seq:
        print(f'{d.shape}')


def print_dict(data:dict):
    for key in data:
        print(f'{key}: {data[key]}')


def resample_to_n(label:np.array, sample_num:int, data_seq):
    # all code increase to certain num (default to max)
    unique_codes, counts = np.unique(label, axis=0, return_counts=True)
    if sample_num <=0:
        sample_num = max(counts)
    new_label = []
    new_data = [[] for d in data_seq]
    for code, count in zip(unique_codes, counts):
        target_row = np.where((label == code).all(axis=1))[0]
        target_row_num = len(target_row)
        for i in range(sample_num):
            random_integer = random.randint(0, target_row_num-1)
            index = target_row[random_integer]
            new_label.append(code)
            for seq_index, seq in enumerate(data_seq):
                new_data[seq_index].append(seq[index])
            # new_label.append(code)
            # new_data.append(pool[random_integer])
    new_label = np.array(new_label)
    for i,d in enumerate(new_data):
        new_data[i] = np.array(d)
    return new_label, new_data


def get_activation_index(data:np.array, on_threshold:int=50,
                         min_off_duration:int=5, min_on_duration:int=5):
    when_on = (data>=on_threshold)
    # find all possible activation window of data, return on-off index
    # Find state changes
    state_changes = np.diff(when_on.astype(np.int8))
    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]
    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        return []
    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return []
    if switch_on_events[-1] > switch_off_events[-1]:
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return []
    assert len(switch_on_events) == len(switch_off_events)
    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = switch_on_events[1:] - switch_off_events[:-1]
        qualified_off_durations_index = np.where(off_durations >= min_off_duration)[0]
        off_index = np.concatenate([qualified_off_durations_index, [len(switch_off_events)-1]])
        switch_off_events = switch_off_events[off_index]
        on_index = np.concatenate([[0], qualified_off_durations_index+1])
        switch_on_events = switch_on_events[on_index]
        assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = off - on
        if duration < min_on_duration:
            continue
        activation = [on, off]
        activations.append(activation)

    activations = np.array(activations)
    return activations


def get_daily_data_from_DataFrame(df_target:pd.DataFrame, day_list:list, start_hour:int=1, end_hour:int=2):
    # get seperated daily (day_list) data from df_target
    # [start_hour]:the start hour of that day; [end_hour]: the end hour of tomorrow
    data_daily_list = []
    for day in day_list:
        today = pd.to_datetime(day) + timedelta(hours=start_hour)
        tomorrow = today + timedelta(hours=24+end_hour)
        df_day = df_target.loc[(df_target.index>today) & (df_target.index<tomorrow)]
        df_day = df_day.reset_index(inplace=False, drop=True)
        df_day = df_day.to_numpy()
        data_daily_list.append(df_day)
    return data_daily_list


def plot_tabular(data:np.array, title:list=None):
    # [tabular data] shape: (batch_n, label_n), with [title] for labels
    col_n = data.shape[-1]
    fig, ax = plt.subplots(col_n, figsize=(5*col_n, 16))
    for i in range(col_n):
        ax[i].plot(data[:,i])
    if title is not None:
        for i in range(col_n):
            ax[i].set_title(title[i])


def get_act_from_daily_data(target_data, target_name, on_threshold, app_min_off_duration):
    # get activation of daily data [target_data], which is from get_daily_data_from_DataFrame()
    # using get_activation_index()
    app_activation_index = {name:np.empty((0,3), dtype=int) for name in target_name}

    for d_index, data_1d in enumerate(target_data):

        for i, name in enumerate(target_name):

            data_1d_app = data_1d[:,i]
            acts = get_activation_index(data_1d_app, on_threshold, app_min_off_duration[name])
            if np.size(acts) != 0:
                tmp_1 = np.full((len(acts), 1), d_index, dtype=int)
                tmp_1 = np.concatenate((tmp_1, acts), axis=-1)
                tmp_2 = app_activation_index[name]
                app_activation_index[name] = np.concatenate((tmp_2, tmp_1), axis=0)

    return app_activation_index


def get_off_state_power(data:np.array, target_name, on_threshold:int=20, int_flag:bool=False):
    # get the average value of [data] below [on_threshold], turn into Int if [int_flag]
    off_state = (data < on_threshold).astype('float')
    data = data * off_state
    off_state_power = np.sum(data, axis=0)/np.sum(off_state, axis=0)
    if int_flag:
        off_state_power = np.floor(off_state_power)
    result = {}
    for i, name in enumerate(target_name):
        result[name] = off_state_power[i]
    return result


def generate_sample_by_label(data_list, app_activation_index, app_name, label, window_l=2000,
                             extend_border_size=5, app_is_periodic=None, off_state_power=None):
    '''
    generate tabular data

    data_list: origin daily data
    app_activation_index: acts of [data_list]
    app_name: for link with other data
    label: the desired label for sample
    app_is_periodic: dictionary of app_name and bool
    '''
    if app_is_periodic is None:
        app_is_periodic = {name:False for name in app_name}

    if off_state_power is None:
         off_state_power = {name:0 for name in app_name}

    shifted_data = np.zeros((window_l, len(app_name)))
    window_l_half = int(window_l/2)

    for i, name in enumerate(app_name):

        shifted_data[:,i] = off_state_power[name]

        if label[i] == 1:

            pool_num = len(app_activation_index[name])

            if app_is_periodic[name]:

                qualified = False

                while not qualified:

                    random_integer = random.randint(0, pool_num-1)
                    d_index, act_s, act_e = app_activation_index[name][random_integer]

                    mid = random.randint(act_s, act_e)
                    tmp_s = mid - window_l_half
                    tmp_e = tmp_s + window_l

                    data_len = len(data_list[d_index][:,i])
                    if (0 <= tmp_s <= data_len) & (0 <= tmp_e <= data_len):
                        qualified = True
                        shifted_data[:,i] = data_list[d_index][tmp_s:tmp_e,i]

            else:

                    random_integer = random.randint(0, pool_num-1)
                    d_index, act_s, act_e = app_activation_index[name][random_integer]
                    act_s -= extend_border_size
                    act_e += extend_border_size

                    tmp_s = window_l_half - random.randint(0, act_e-act_s)
                    tmp_e = tmp_s + act_e - act_s

                    shifted_data[tmp_s:tmp_e,i] = data_list[d_index][act_s:act_e,i]

        elif app_is_periodic[name]:

            qualified = False
            pool_num = len(app_activation_index[name])

            while not qualified:

                random_integer = random.randint(0, pool_num-2)
                d_index_1, _, act_e = app_activation_index[name][random_integer]
                d_index_2, act_s, _ = app_activation_index[name][random_integer+1]

                if d_index_1!=d_index_2:
                    continue
                else:
                    mid = random.randint(act_e, act_s)
                    tmp_s = mid - window_l_half
                    tmp_e = tmp_s + window_l

                    data_len = len(data_list[d_index_1][:,i])
                    if (0 <= tmp_s <= data_len) & (0 <= tmp_e <= data_len):
                        qualified = True
                        shifted_data[:,i] = data_list[d_index_1][tmp_s:tmp_e,i]

    return shifted_data


def align_seq(seq_1:np.array, seq_2:np.array, shift_1_to_2:int):
    # align two seq to same length
    len_1 = len(seq_1)
    len_2 = len(seq_2)
    # seq 1
    a, b = 0, len_1
    c, d = shift_1_to_2, shift_1_to_2+len_2
    start = max(a, c)
    end = min(b, d)
    new_seq_1 = seq_1[start:end]
    # seq 2
    a, b = 0, len_2
    c, d = -shift_1_to_2, len_1 - shift_1_to_2
    start = max(a, c)
    end = min(b, d)
    new_seq_2 = seq_2[start:end]
    return new_seq_1, new_seq_2


def generate_window_samples(input_1:np.array, input_2:np.array, window_size_1:int, window_size_2:int, offset:int=0):
    # input_1 and input_2 supposed to be two or less dim, the first dim being time, the second dim being appliance
    len_1 = len(input_1)
    len_2 = len(input_2)
    if len_1 != len_2:
        print('Length is equal: {}, {}'.format(len_1, len_2))
        return

    window_num = len_1 - max(window_size_1, window_size_2+offset) + 1

    output_1 = []
    output_2 = []

    for i in range(window_num):
        window_1_start = i
        window_1_end = i + window_size_1
        tmp_window_1 = input_1[window_1_start:window_1_end]
        window_2_start = i + offset
        window_2_end = i + offset + window_size_2
        tmp_window_2 = input_2[window_2_start:window_2_end]
        output_1.append(tmp_window_1)
        output_2.append(tmp_window_2)

    output_1 = np.array(output_1)
    output_2 = np.array(output_2)

    output_1 = output_1.squeeze()
    output_2 = output_2.squeeze()

    return output_1, output_2



def calculate_improvement_ratio(vanilla_score, neo_score, psr=1):
    # psr = performace_score_relation 
    if isinstance(neo_score, (list, tuple)):
        improvement_ratio = [(neo - vanilla_score) / vanilla_score for neo in neo_score]
        if psr>0:
            return improvement_ratio
        else:
            return [element * -1 for element in improvement_ratio]
    else:
        improvement_ratio = (neo_score - vanilla_score) / vanilla_score
        if psr>0:
            return improvement_ratio
        else:
            return improvement_ratio * -1



def get_activation_index_from_states(data:np.array, min_off_duration:int=5, min_on_duration:int=5):
    # find all possible activation window of data, return on-off index
    # Find state changes
    pre_0_len = min_off_duration + min_on_duration
    post_0_len = min_off_duration + min_on_duration
    data = np.concatenate([np.zeros(pre_0_len),np.squeeze(data), np.zeros(post_0_len)])
    state_changes = np.diff(data.astype(np.int8))
    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]
    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        return []
    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return []
    if switch_on_events[-1] > switch_off_events[-1]:
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return []
    assert len(switch_on_events) == len(switch_off_events)
    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = switch_on_events[1:] - switch_off_events[:-1]
        qualified_off_durations_index = np.where(off_durations >= min_off_duration)[0]
        off_index = np.concatenate([qualified_off_durations_index, [len(switch_off_events)-1]])
        switch_off_events = switch_off_events[off_index]
        on_index = np.concatenate([[0], qualified_off_durations_index+1])
        switch_on_events = switch_on_events[on_index]
        assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = off - on
        if duration < min_on_duration:
            continue
        activation = [on+1-pre_0_len, off+1-pre_0_len]
        activations.append(activation)

    activations = np.array(activations)
    return activations
