from dgl.data import FraudYelpDataset, FraudAmazonDataset, TolokersDataset
from dgl.data.utils import load_graphs
import dgl
import torch

class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None

        if name == 'tfinance':
            graph, _ = load_graphs('dataset/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

        elif name == 'tsocial':
            graph, _ = load_graphs('dataset/tsocial')
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)

        elif name == 'tolokers':
            dataset = TolokersDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
            graph.ndata['feature'] = graph.ndata.pop('feat')

        elif name == 'elliptic':
            graph, _ = load_graphs('dataset/elliptic')
            graph = graph[0]
            if len(graph.ndata['label'].shape) > 1 and graph.ndata['label'].shape[1] > 1:
                graph.ndata['label'] = graph.ndata['label'].argmax(1)
        else:
            print('No such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph
