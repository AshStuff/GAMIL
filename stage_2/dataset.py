import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, data, num_graphs, num_nodes_per_graph, distance):
        super().__init__()
        self.data = data
        self.num_graphs = num_graphs
        self.num_nodes_per_graph = num_nodes_per_graph
        self.distance = distance

    def __getitem__(self, item):
        data = self.data[item]
        class_name = int(os.path.dirname(data).split('/')[-1])
        data_np = np.load(data)
        features = data_np["features"]
        coords = data_np["coords"]
        all_data = []
        for k in range(self.num_graphs):
            sample = np.random.choice(np.arange(0, len(features)), size=self.num_nodes_per_graph)
            sample_features = features[sample]
            sample_coords = coords[sample]
            adj = self.create_adjacency_matrix(sample_coords, self.distance)
            all_data.append((torch.from_numpy(sample_features).float(), torch.from_numpy(adj).float()))
        return {"data": all_data, "label": class_name}

    def create_adjacency_matrix(self, coords, d):
        arr_x = (coords[:, 0, np.newaxis].T - coords[:, 0, np.newaxis]) ** 2
        arr_y = (coords[:, 1, np.newaxis].T - coords[:, 1, np.newaxis]) ** 2
        arr = np.sqrt(arr_x + arr_y)
        arr[arr > d] = 0
        arr[arr <= d] = 1
        return arr

    def __len__(self):
        return len(self.data)
