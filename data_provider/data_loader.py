import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class TrajDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.__read_data__()

    def __get_traffic_seq(self):
        edge_df = self.edge_df
        traj_df = self.traj_df

        traffic_df = pd.DataFrame(traj_df[['EdgeID', 'Time', 'Hour']])  # deep copy
        traffic_df['TimeDiff'] = traffic_df["Time"].apply(lambda x: np.diff(x))
        traffic_df['Hour'] = traffic_df['Hour'].apply(lambda x: np.array(x[:-1]))
        traffic_df = traffic_df[['EdgeID', 'TimeDiff', 'Hour']].explode(['EdgeID', 'TimeDiff', 'Hour'])
        traffic_df['EdgeID'] = traffic_df['EdgeID'].astype(int)
        traffic_df['TimeDiff'] = traffic_df['TimeDiff'].astype(float)
        traffic_df['Hour'] = traffic_df['Hour'].astype(int)
        traffic_df = traffic_df.merge(
            edge_df[['EdgeID', 'Length']],
            on='EdgeID',
            how='left'
        )
        traffic_df = traffic_df.groupby(['EdgeID', 'Hour']).agg(
            VehicleNum=('TimeDiff', 'size')
        ).reset_index()
        traffic_df['NormVehicleNum'] = (traffic_df['VehicleNum'] - traffic_df['VehicleNum'].mean()) / traffic_df['VehicleNum'].std()

        traj_df = pd.DataFrame(traj_df[['EdgeID', 'Hour']])
        traj_df['TrajID'] = traj_df.index
        traj_df['Hour'] = traj_df['Hour'].apply(lambda x: np.array(x[:-1]))
        traj_df = traj_df.explode(['EdgeID', 'Hour'])
        traj_df = traj_df.merge(
            traffic_df,
            on=['EdgeID', 'Hour'],
            how='left'
        )
        traj_df = traj_df.groupby('TrajID', as_index=True).agg(
            EdgeID=('EdgeID', list),
            Hour=('Hour', list),
            VehicleNum=('VehicleNum', list),
            NormVehicleNum=('NormVehicleNum', list)
            )
        
        traffic_seq = traj_df['NormVehicleNum'].apply(lambda x: torch.tensor(x, dtype=torch.float32)).values
        return traffic_seq

    def __read_data__(self):
        self.edge_df = pd.read_parquet(os.path.join(self.root_path, 'roadnet', 'edge.parquet'))
        self.traj_df = pd.read_parquet(os.path.join(self.root_path, 'trajs', 'traj.parquet'))

        edge_df = self.edge_df
        traj_df = self.traj_df

        # 1. get edge sequence
        self.edge_seq = traj_df['EdgeID'].apply(lambda x: torch.tensor(x)).values

        # 2. get edge features
        edge_df['ClassIndex'], _labels = pd.factorize(edge_df['Class'])
        # maybe avg speed later...
        self.edge_features = edge_df[['ClassIndex', 'Length']].values
        self.real_time_traffic = True
        if self.real_time_traffic:
            self.traffic_seq = self.__get_traffic_seq()

        self.edge_features = torch.tensor(self.edge_features, dtype=torch.float32)
        mean = self.edge_features.mean(dim=0)
        std = self.edge_features.std(dim=0)
        self.edge_features = (self.edge_features - mean) / std

        # 3. get time features
        traj_df['Time'] = traj_df['Time'] / 86399 - 0.5
        traj_df['Hour'] = traj_df['Hour'] / 23 - 0.5
        traj_df['Minute'] = traj_df['Minute'] / 59 - 0.5
        traj_df['Second'] = traj_df['Second'] / 59 - 0.5
        self.timeF = traj_df.apply(lambda row: torch.tensor(list(zip(row['Time'], row['Hour'], row['Minute'], row['Second'])), dtype=torch.float32), axis=1).values

    def __len__(self):
        return len(self.timeF)

    def __getitem__(self, idx):
        if self.real_time_traffic:
            edge_features = torch.cat([self.edge_features[self.edge_seq[idx]], self.traffic_seq[idx].unsqueeze(1)], dim=1)
        else:
            edge_features = self.edge_features[self.edge_seq[idx]]
        return self.edge_seq[idx], edge_features, self.timeF[idx][:-1, :], self.timeF[idx][1:, 0]

if __name__ == "__main__":
    root_path = 'dataset/processed/shenzhen_20201104'
    traj = TrajDataset(root_path)
    for i in range(10):
        print(traj[i])
        exit()