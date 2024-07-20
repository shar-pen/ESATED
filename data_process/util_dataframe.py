import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def resample_Dataframe(input:pd.DataFrame, sample_seconds=1) -> pd.DataFrame:
    return input.resample((str(sample_seconds)+'S')).min().\
            fillna(method='backfill',limit=1).dropna()


def plot_multi_cols(plot_df:pd.DataFrame, window_l:int, window_s:int=0, xaxis_on:bool=True):
    if window_l<=0:
        window_l = plot_df.__len__()-window_s
    cols = plot_df.columns.tolist()
    fig, axs = plt.subplots(len(cols))
    for i in range(len(cols)):
        axs[i].plot(plot_df.loc[window_s:window_s+window_l, cols[i]].to_numpy())
        axs[i].set_title(cols[i])
        if not xaxis_on:
            axs[i].get_xaxis().set_visible(False)


def get_date(df:pd.DataFrame):
    df_daily = df.resample('D').sum()
    time_list = df_daily.index.strftime('%Y-%m-%d').tolist()
    return time_list
