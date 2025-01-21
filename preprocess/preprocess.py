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
        self.processed_folder = 'dataset/processed/' + self.data

        self.spark = SparkSession.builder \
            .appName("Preprocess") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

    def run(self): 
        self.__read_roadnet()
        self.preprocess_roadnet()

        self.__read_trajs()
        self.preprocess_trajs()
        self.save()

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
        df = rdd_with_index.map(lambda x: (x[1], *x[0])).toDF(["EdgeID"] + self.edge_df.columns)
        df = df.withColumn("Origin", df["Origin"].cast("int")) \
                .withColumn("Destination", df["Destination"].cast("int"))
        df = df.withColumn("Length", df["Length"].cast("double"))

        df = df.withColumn('Geometry', F.split(df['Geometry'], "[_\\-]"))
        df = df.withColumn("Longitude", F.expr("filter(Geometry, (x, i) -> i % 2 == 0)").cast("array<double>"))
        df = df.withColumn("Latitude", F.expr("filter(Geometry, (x, i) -> i % 2 == 1)").cast("array<double>"))
        df = df.drop("Geometry")
        self.edge_df = df

    def preprocess_trajs(self):
        df = self.traj_df
        df = df.withColumn('VehicleID', df["VehicleID"].cast('int'))
        df = df.withColumn('TripID', df["TripID"].cast('int'))
        df = df.withColumn('Length', df["Length"].cast('double'))
        df = df.withColumn('Duration', df["Duration"].cast('double'))
        df = df.withColumn('DepartureTime', df["DepartureTime"].cast('double'))

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
        df = df.withColumn("Nodes", F.expr("filter(Points, (x, i) -> i % 2 == 0)").cast("array<int>"))

        rdd = df.rdd.map(lambda row: row.asDict())
        rdd = rdd.map(lambda row: {**row, "Edge_ID": [node2edge.value.get(node_pair, -1) for node_pair in zip(row["Nodes"][:-1], row["Nodes"][1:])]})
        df = rdd.toDF()
        # old implementation using udf
        # map_udf = F.udf(lambda arr: [node2edge.value.get(node_pair, -1) for node_pair in zip(arr[:-2], arr[1:])], ArrayType(IntegerType()))
        # df = df.withColumn("Edge_ID", map_udf("Nodes"))  
        df = df.filter(~F.array_contains(F.col("Edge_ID"), -1))
        
        # 2.2 map Time (Node) to Time (Edge)
        df = df.withColumn("Time", F.expr("filter(Points, (x, i) -> i % 2 == 1)").cast("array<double>")) 
        df = df.withColumn("Time_Diff", F.expr("transform(slice(Time, 1, size(Time) - 1), (x, i) -> Time[i + 1]-x)"))
        # df = df.withColumn("Time", F.expr("slice(Time, 2, size(Time)-1)"))
        df = df.withColumn("Hour", F.expr("transform(Time, x -> cast(x / 3600 as int) % 24)"))
        df = df.withColumn("Minute", F.expr("transform(Time, x -> cast(x % 3600 / 60 as int))"))
        df = df.withColumn("Second", F.expr("transform(Time, x -> cast(x % 60 as int))"))

        # 2.3 compute Speed On Edge
        # To Do
        # df = df.withColumn("Speed", F.expr("transform(Time_Diff, (x, i) -> x / Time_Diff[i])"))
        
        # df = df.drop("Destinations", 'Zipped', 'Points', 'Time_Diff')
        df = df.drop("Destinations", 'Zipped', 'Points', 'Time_Diff')
        self.traj_df = df
        # to do 
        # split the meta data and seq data
        # return pos_traj, time_traj
        return

    def get_node2edge(self):
        node2edge = self.edge_df.rdd.map(lambda row: ((row["Origin"], row["Destination"]), row["EdgeID"])).collectAsMap()
        return node2edge

    def save(self):
        # save trajs as parquet
        trajs_path = os.path.join(self.processed_folder, 'trajs', 'traj.parquet')
        if not os.path.exists(trajs_path):
            os.makedirs(trajs_path)
        self.traj_df.write.mode('overwrite').parquet(trajs_path)

        # Save 
        edge_path = os.path.join(self.processed_folder, 'roadnet', 'edge.parquet')
        if not os.path.exists(edge_path):
            os.makedirs(edge_path)
        self.edge_df.write.mode('overwrite').parquet(edge_path)

        # Save edgelist.txt
        edgelist_path = os.path.join(self.processed_folder, 'roadnet', 'edgelist.txt')
        edgelist_df = self.edge_df.select('Origin', 'Destination').toPandas()
        edgelist_df.to_csv(edgelist_path, sep=" ", header=False, index=False)

        self.traj_df.show()
        self.edge_df.show()
        print(self.traj_df)
        print(self.edge_df)

if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.run()