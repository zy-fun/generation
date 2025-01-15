import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, IntegerType
import time
# from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.datasets import load_from_tsfile_to_dataframe
# import warnings
# from utils.augmentation import run_augmentation_single

class TrajDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.spark = SparkSession.builder \
            .appName("Preprocess") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.__read_data__()

    def __read_data__(self):
        self.edge_df = self.spark.read.parquet(os.path.join(self.root_path, 'roadnet', 'edge.parquet'))
        self.traj_df = self.spark.read.parquet(os.path.join(self.root_path, 'trajs', 'traj.parquet'))

        edge_df = self.edge_df
        traj_df = self.traj_df

        # scale time features
        traj_df = traj_df.withColumn("Time", F.expr("transform(Time, x -> x / 86399 - 0.5)"))
        traj_df = traj_df.withColumn("Hour", F.expr("transform(Hour, x -> x / 23 - 0.5)"))
        traj_df = traj_df.withColumn("Minute", F.expr("transform(Minute, x -> x / 59 - 0.5)"))
        traj_df = traj_df.withColumn("Second", F.expr("transform(Second, x -> x / 59 - 0.5)"))
        # traj_df = traj_df.drop("Time", "Hour", "Minute", "Second")
        
        # print(traj_df.select('TimeF').toPandas()['TimeF'].tolist())
        # self.timeF = traj_df.rdd.map(lambda row: list(map(list, zip(row["Time"], row["Hour"], row["Minute"], row["Second"])))).collect()
        self.timeF = traj_df.rdd.map(lambda row: list(zip(row["Time"], row["Hour"], row["Minute"], row["Second"]))).collect()
        self.timeF = [torch.tensor(x) for x in self.timeF]
        self.spark.stop()

    def __len__(self):
        return len(self.timeF)

    def __getitem__(self, idx):
        return [], self.timeF[idx]

if __name__ == "__main__":
    root_path = 'dataset/processed/shenzhen_20201104'
    traj = TrajDataset(root_path)
    for i in range(10):
        print(traj[i])
        exit()