import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.feature import StringIndexer
import time
from tqdm import tqdm

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

        # 1. get edge sequence
        self.edge_seq = traj_df.select("Edge_ID").rdd.map(lambda row: row["Edge_ID"]).collect()
        self.edge_seq = [torch.tensor(x) for x in self.edge_seq]

        # 2. get edge features
        indexer = StringIndexer(inputCol="Class", outputCol="ClassIndex")
        edge_df = indexer.fit(edge_df).transform(edge_df)
        # maybe avg speed later...
        self.edge_features = edge_df.select("ClassIndex", "Length").rdd.map(lambda row: (row["ClassIndex"], row["Length"])).collect()
        self.edge_features = torch.tensor(self.edge_features)
        mean = self.edge_features.mean(dim=0)
        std = self.edge_features.std(dim=0)
        self.edge_features = (self.edge_features - mean) / std

        # 3. get time features
        traj_df = traj_df.withColumn("Time", F.expr("transform(Time, x -> x / 86399 - 0.5)"))
        traj_df = traj_df.withColumn("Hour", F.expr("transform(Hour, x -> x / 23 - 0.5)"))
        traj_df = traj_df.withColumn("Minute", F.expr("transform(Minute, x -> x / 59 - 0.5)"))
        traj_df = traj_df.withColumn("Second", F.expr("transform(Second, x -> x / 59 - 0.5)"))



        self.timeF = traj_df.select("Time", "Hour", "Minute", "Second") \
            .rdd.map(lambda row: list(zip(row["Time"], row["Hour"], row["Minute"], row["Second"]))).collect()
        self.timeF = [torch.tensor(x) for x in self.timeF]

        self.spark.stop()

    def __len__(self):
        return len(self.timeF)

    def __getitem__(self, idx):
        return self.edge_seq[idx], self.edge_features[self.edge_seq[idx]], self.timeF[idx][:-1, :], self.timeF[idx][1:, 0]

if __name__ == "__main__":
    root_path = 'dataset/processed/shenzhen_20201104'
    traj = TrajDataset(root_path)
    for i in range(10):
        print(traj[i])
        exit()