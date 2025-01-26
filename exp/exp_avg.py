import os
from data_provider.data_factory import root_dict, split_indices
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from utils.metrics import metric

class Exp_AVG(object):
    def __init__(self, args):
        self.args = args
        self.__read_data__()

    def __read_data__(self):
        self.root_path = root_dict[self.args.data]
        edge_df = pd.read_parquet(os.path.join(self.root_path, 'roadnet', 'edge.parquet'))
        traj_df = pd.read_parquet(os.path.join(self.root_path, 'trajs', 'traj.parquet'))

        self.traj_df = traj_df[['EdgeID', 'Time', 'DepartureTime', 'Hour']]
        self.edge_df = edge_df

        train_indices, val_indices, test_indices = split_indices(len(self.traj_df))
        self.train_data = self.traj_df.iloc[train_indices]
        self.val_data = self.traj_df.iloc[val_indices]
        self.test_data = self.traj_df.iloc[test_indices]

    def vali(self, data='val'):
        if data == 'val':
            vali_data = self.val_data
        elif data == 'test':
            vali_data = self.test_data
        else:
            vali_datar = self.train_data
        pass

    def train(self):
        df = self.train_data

        df["TimeDiff"] = df["Time"].apply(lambda x: np.diff(x))
        df['Hour'] = df['Hour'].apply(lambda x: np.array(x[:-1]))
        df = df[['EdgeID', 'TimeDiff', 'Hour']].explode(['EdgeID', 'TimeDiff', 'Hour'])
        df['EdgeID'] = df['EdgeID'].astype(int)
        df['TimeDiff'] = df['TimeDiff'].astype(float)
        df['Hour'] = df['Hour'].astype(int)
        df = df.merge(
            self.edge_df[['EdgeID', 'Length']],
            on='EdgeID',
            how='left'
        )

        df['Speed'] = df['Length'] / df['TimeDiff']
        df = df[~np.isinf(df['Speed'])]

        self.global_avg_speed = df['Speed'].mean()

        # df = df.groupby('EdgeID').agg(
        #     Speed=('Speed', 'mean'),
        #     Count=('Speed', 'size')
        # ).reset_index()

        self.length_dict = self.edge_df.set_index('EdgeID')['Length'].to_dict()
        self.time_dict = df.set_index('EdgeID')['TimeDiff'].to_dict()
        self.time_dicts_of_hour = df.groupby('Hour').apply(lambda x: x.set_index('EdgeID')['TimeDiff'].to_dict())

    def test(self):
        df = self.test_data
        by_hour = self.args.avg_by_hour

        if by_hour:
            df['PredTimeDiff'] = df.apply(lambda row: np.array([self.time_dicts_of_hour[row['Hour'][0]].get(i, 
                self.time_dict.get(i, self.length_dict[i] / self.global_avg_speed)) for i in row['EdgeID']]), axis=1)
        else:
            df['PredTimeDiff'] = df["EdgeID"].apply(lambda x: np.array([self.time_dict.get(i, self.length_dict[i] / self.global_avg_speed) for i in x]))
        df['Pred'] = df.apply(lambda row: np.cumsum(np.insert(row['PredTimeDiff'], 0, row['DepartureTime'])), axis=1)

        preds = df['Pred'].tolist()
        trues = df['Time'].tolist()

        print(preds[-1])
        print(trues[-1])
        result = metric(preds, trues)
        for k, value in result.items():
            print(f'{k}: {value}')

        preds = [x[-1:] for x in preds]
        trues = [x[-1:] for x in trues]
        result = metric(preds, trues)
        for k, value in result.items():
            print(f'{k}: {value}')
        return
