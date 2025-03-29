import pandas as pd
import os

# data transformation for cblab

root_path = "./dataset/raw/shenzhen"

node_df = pd.read_csv(os.path.join(root_path, 'roadnet', 'node_shenzhen.csv'))
node_df = node_df[['NodeID', 'Longitude', 'Latitude']]
if not os.path.exists('./dataset/cblab'):
    os.mkdir('./dataset/cblab')
node_df.to_csv('./dataset/cblab/node.csv', index=False, header=False)

edge_df = pd.read_csv(os.path.join(root_path, 'roadnet', 'edge_shenzhen.csv'))
edge_df['EdgeID'] = range(len(edge_df))
edge_df[['EdgeID', 'Origin', 'Destination']].to_csv('./dataset/cblab/edge1.csv', index=False, header=False)
edge_df[['EdgeID', 'Class']].to_csv('./dataset/cblab/edge2.csv', index=False, header=False)