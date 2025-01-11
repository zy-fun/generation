from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, IntegerType
import os
from tqdm import tqdm


class Preprocess:
    def __init__(self, data='shenzhen_20201104'):
        self.data = data

        self.city = 'shenzhen' if 'shenzhen' in self.data else 'jinan'
        self.raw_folder = 'dataset/raw/' + self.city
        self.processed_folder = 'dataset/processed/' + self.city

        self.spark = SparkSession.builder \
            .appName("Preprocess") \
            .getOrCreate()
        
        self.__read_roadnet()
        self.preprocess_roadnet()

        self.__read_trajs()
        self.preprocess_trajs()

    def __read_roadnet(self):
        self.edge_path = os.path.join(self.raw_folder, 'roadnet', 'edge_' + self.city + '.csv')
        self.edge_df = self.spark.read.csv(self.edge_path, header=True)
        self.num_edges = self.edge_df.count()

        self.node_path = os.path.join(self.raw_folder, 'roadnet', 'node_' + self.city + '.csv')
        self.node_df = self.spark.read.csv(self.node_path, header=True)

    def __read_trajs(self):
        self.traj_path = os.path.join(self.raw_folder, 'trajs', 'traj_' + self.data + '.csv')
        self.traj_df = self.spark.read.csv(self.traj_path, header=True)

    def preprocess_roadnet(self):
        rdd_with_index = self.edge_df.rdd.zipWithIndex()
        self.edge_df = rdd_with_index.map(lambda x: (x[1], *x[0])).toDF(["EdgeID"] + self.edge_df.columns)
        self.edge_df = self.edge_df.withColumn("Origin", self.edge_df["Origin"].cast("int")) \
                .withColumn("Destination", self.edge_df["Destination"].cast("int"))

        pass

    def preprocess_trajs(self):
        df = self.traj_df

        # 1. filter trajs
        # filter trajs with length < 1 km
        df = df.filter(df['Length'] >= 2000) 
        df = df.filter(df['Length'] <= 20000)

        # filter trajs with speed > 120 km/h
        df = df.withColumn('Avgspeed', df['Length'] / df['Duration'] * 3.6)
        df = df.filter(df['Avgspeed'] <= 80)
        df = df.filter(df['Avgspeed'] >= 10)
        
        # filter trajs with duration
        df = df.filter(df['Duration'] / 3600 <= 1)
        df = df.filter(df['Duration'] / 60 >= 5)

        # 2. map (NodeID, Time) sequence to (EdgeID, Time) sequence
        node2edge = self.get_node2edge()
        node2edge = self.spark.sparkContext.broadcast(node2edge)
        # 2.1 map NodeID to EdgeID
        df = df.withColumn('Points', F.split(df['Points'], "[_\\-]"))
        df = df.withColumn("Origins", F.expr("filter(Points, (x, i) -> i % 2 == 0)").cast("array<int>"))
        df = df.withColumn("Destinations", F.expr("slice(Origins, 2, size(Origins)-1)"))
        df = df.withColumn("Zipped", F.arrays_zip("Origins", "Destinations"))
        map_udf = F.udf(lambda arr: [node2edge.value.get(node_pair, -1) for node_pair in arr], ArrayType(IntegerType()))
        df = df.withColumn("Edge_ID", map_udf("Zipped"))
        
        # 2.2 map Time (Node) to Time (Edge)
        df = df.withColumn("Time", F.expr("filter(Points, (x, i) -> i % 2 == 1)").cast("array<double>")) 
        df = df.withColumn("Time_Diff", F.expr("transform(slice(Time, 1, size(Time) - 1), (x, i) -> Time[i + 1]-x)"))
        df = df.withColumn("Time", F.expr("slice(Time, 2, size(Time)-1)"))

        # 2.3 compute Speed On Edge
        # To Do
        # df = df.withColumn("Speed", F.expr("transform(Time_Diff, (x, i) -> x / Time_Diff[i])"))
        
        df = df.drop("Origins", "Destinations", 'Zipped', 'Points', 'Time_Diff')
        # to do 
        # split the meta data and seq data
        # return pos_traj, time_traj

    def get_node2edge(self):
        node2edge = self.edge_df.rdd.map(lambda row: ((row["Origin"], row["Destination"]), row["EdgeID"])).collectAsMap()
        return node2edge

    def save(self, data):
        # Save data
        pass

    def run(self):
        # Load data
        data = self.spark.load_data(self.data_path)
        # Preprocess data
        data = self.preprocess(data)
        # Save data
        self.save(data)

if __name__ == '__main__':
    preprocess = Preprocess()