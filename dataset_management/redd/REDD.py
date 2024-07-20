import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from dataset_management.functions import unique_order_preserved, concatenate_dataframes_to_one


class REDD():

    def __init__(self, in_path, out_path) -> None:
        self.data_directory = in_path
        self.save_directory = out_path

    def to_Dataframe_V1(self, Houses:list, Appliances:list, sample_seconds=1, only_first:bool=True, summed:bool=False):
        start_time = time.time()
        # house
        for h in Houses:
            print('House:', h)
            house_df = pd.DataFrame()
            house_path = os.path.join(self.data_directory, 'house_'+str(h))
            labels = pd.read_csv(os.path.join(house_path, 'labels.dat'),header=None,delimiter=' ',
                                 usecols=[0,1], names=['channel', 'appliance_name'])
            # appliance
            if Appliances == []:
                Appliances = set(labels['appliance_name'])
            for app in Appliances:
                channels = labels.loc[labels['appliance_name']==app, 'channel'].tolist()
                if only_first:
                    channels = [channels[0]]
                print('    Channels of {}: {}'.format(str(app), channels))
                # channel
                app_df = pd.DataFrame()
                for i,c in enumerate(channels):
                    file_name = 'channel_' + str(c) + '.dat'
                    tmp_df = pd.read_table(os.path.join(house_path, file_name), sep="\s+",
                                            usecols=[0, 1], names=['time', f'{app}_{i}'], dtype={'time': str})
                    tmp_df['time'] = pd.to_datetime(tmp_df['time'], unit='s')
                    tmp_df.set_index('time', inplace=True)
                    app_df = app_df.join(tmp_df, how='outer')
                if summed:
                    house_df[str(app)] = app_df.sum(axis=1)
                else:
                    house_df = house_df.join(app_df, how='outer')
            house_df_resample = house_df.resample((str(sample_seconds)+'S')).max().\
                fillna(method='backfill',limit=1).dropna()
            house_df_resample.to_csv(os.path.join(self.save_directory, 'house_'+str(h)+'.csv'),
                                   index=True, header=True)
        print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
        print("Please find files in: ", self.save_directory)


    def to_Dataframe_V2(self, Houses:list=[1,2,3,4,5,6], sample_seconds=1):
        start_time = time.time()
        # house
        for h in Houses:

            house_df = pd.DataFrame()
            house_path = os.path.join(self.data_directory, 'house_'+str(h))
            labels = pd.read_csv(os.path.join(house_path, 'labels.dat'),header=None,delimiter=' ',
                                 usecols=[0,1], names=['channel', 'appliance_name'])
            len = labels.__len__()

            print(f'House: {h} , Channels: {len}')
            # labels['appliance_name'] = labels['appliance_name'] + '-' + labels['channel'].astype(str)
            # Appliances = labels['appliance_name']
            # Channels = labels['channel']
            df_list = []
            for index, row in labels.iterrows():

                print(f'{index+1}', end='.')
                c = row['channel']
                app = row['appliance_name']

                file_name = r'channel_' + str(c) + '.dat'
                tmp_df = pd.read_table(os.path.join(house_path, file_name), sep="\s+",
                            usecols=[0, 1], names=['time', f'{app}-{c}'], dtype={'time': str})
                tmp_df['time'] = pd.to_datetime(tmp_df['time'], unit='s')
                tmp_df.set_index('time', inplace=True)
                df_list.append(tmp_df)

            house_df = concatenate_dataframes_to_one(df_list)
            house_df_resample = house_df.resample((str(sample_seconds)+'S')).max().fillna(0)
            house_df_resample.to_csv(os.path.join(self.save_directory, 'house_'+str(h)+'.csv'),
                                   index=True, header=True)
            print()

        print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
        print("Please find files in: ", self.save_directory)

