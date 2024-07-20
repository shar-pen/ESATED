import pandas as pd


def load_dataframe(directory, building, channel, col_names=['time', 'data'], nrows=None):
    df = pd.read_table(directory + 'house_' + str(building) + '/' + 'channel_' +
                       str(channel) + '.dat',
                       sep="\s+",
                       nrows=nrows,
                       usecols=[0, 1],
                       names=col_names,
                       dtype={'time': str},
                       )
    return df

def unique_order_preserved(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def concatenate_dataframes(dfs:list):
    new_dataframes = []
    for i in range(0, len(dfs)-1, 2):
        df_concatenated = dfs[i].join(dfs[i+1], how='outer').fillna(0)
        new_dataframes.append(df_concatenated)
    if len(dfs)%2==1:
        new_dataframes.append(dfs[-1])
    return new_dataframes

def concatenate_dataframes_to_one(dfs:list):
    while len(dfs)>1:
        dfs = concatenate_dataframes(dfs)
    return dfs[0]