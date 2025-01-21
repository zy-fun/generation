import numpy as np
import os

def load_embeddings(root_path, emb_type='cat'):
    emb_path = os.path.join(root_path, 'roadnet', 'nodes_embeddings.npy')
    edge_list_path = os.path.join(root_path, 'roadnet', 'edgelist.txt')
    nodes_embeddings = np.load(emb_path)
    edgelist = np.loadtxt(edge_list_path, dtype=int).T
    if emb_type == 'cat':
        edge_embeddings = np.concatenate([nodes_embeddings[edgelist[0]], nodes_embeddings[edgelist[1]]], axis=-1)
    else:
        raise NotImplementedError
    return edge_embeddings