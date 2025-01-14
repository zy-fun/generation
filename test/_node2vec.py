import os
import networkx as nx
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from node2vec import Node2Vec

# path param
data = 'shenzhen_20201104'
city = 'shenzhen' if 'shenzhen' in data else 'jinan'
raw_folder = 'dataset/processed/' + city
processed_folder = 'dataset/processed/' + city

edge_path = os.path.join(raw_folder, 'roadnet', 'edge_' + city + '.parquet')
edge_df = pd.read_parquet(edge_path)
edgelist_path = os.path.join(processed_folder, 'roadnet', 'edgelist.txt')

# node2vec param
dim = 64
p = 10
q = 1

# cluster param / visualization param
n_clusters = 20


# 
G = nx.read_edgelist(edgelist_path, nodetype=int, create_using=nx.DiGraph())
edgelist = np.array(list(zip(*G.edges)), dtype=np.int64)
node2vec = Node2Vec(G, dimensions=dim, walk_length=30, num_walks=10, workers=8, p=p)
model = node2vec.fit(window=10, min_count=1, batch_words=10000, workers=8)
embeddings = model.wv

nodes_embeddings = np.zeros((G.number_of_nodes(), dim))
for node in embeddings.index_to_key:
    nodes_embeddings[int(node)] = embeddings[node]

embed_path = os.path.join(processed_folder, 'roadnet', 'nodes_embeddings.npy')
np.save(embed_path, nodes_embeddings)
print(nodes_embeddings.shape)

edge_embeddings = np.concatenate([nodes_embeddings[edgelist[0]], nodes_embeddings[edgelist[1]]], axis=-1)
print(edge_embeddings.shape)

# cluster
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(edge_embeddings)

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(edge_embeddings)

sns.set(style="whitegrid")
palette = sns.color_palette("hsv", n_colors=n_clusters)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=cluster_labels,
    palette=palette,
    s=100,
    edgecolor='k',
    legend='full'
)

plt.title("t-SNE Visualization of Clustered Embeddings", fontsize=16)
plt.xlabel("t-SNE Dimension 1", fontsize=14)
plt.ylabel("t-SNE Dimension 2", fontsize=14)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('results/edge_embeddings.png')
plt.close()

#######################################

palette = sns.color_palette("hsv", n_clusters)  
cluster_colors = {cluster: palette[i] for i, cluster in enumerate(range(n_clusters))}

plt.figure(figsize=(10, 8))

for cluster, color in tqdm(cluster_colors.items()):
    cluster_index = np.where(cluster_labels == cluster)[0]
    if len(cluster_index) == 0:
        continue
    cluster_edges = edge_df.iloc[cluster_index]
    
    lines = []
    for i, row in cluster_edges.iterrows():
        lon = row['Longitude']
        lat = row['Latitude']
        lines.append(list(zip(lon, lat)))
        
    print(len(lines))
    lc = LineCollection(lines, colors=[color], linewidths=2, label=f'Cluster {cluster}')
    
    plt.gca().add_collection(lc)

all_lon = edge_df['Longitude'].explode().array
all_lat = edge_df['Latitude'].explode().array
plt.xlim(min(all_lon), max(all_lon))
plt.ylim(min(all_lat), max(all_lat))

plt.title("Road Network with Clustered Edges", fontsize=16)
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)

plt.legend()

plt.grid(True)
plt.tight_layout()
plt.savefig('results/edges_cluster.png')