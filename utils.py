from copy import deepcopy
from numbers import Number
from torch.autograd import Variable
from texttable import Texttable
from param_parser import parameter_parser
import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, subgraph
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T

VERY_SMALL_NUMBER = 1e-12
np.random.seed(12345)


def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False):
    if name in ["IMDB-BINARY", "REDDIT-BINARY", "COLLAB", "IMDB-MULTI"]:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
        dataset = TUDataset(path, name, cleaned=cleaned)
        dataset.data.edge_attr = None

        if dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)

        if not sparse:
            num_nodes = max_num_nodes = 0
            for data in dataset:
                num_nodes += data.num_nodes
                max_num_nodes = max(data.num_nodes, max_num_nodes)

            # Filter out a few really large graphs in order to apply DiffPool.
            if name == 'REDDIT-BINARY':
                num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
            else:
                num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

            indices = []
            for i, data in enumerate(dataset):
                if data.num_nodes <= num_nodes:
                    indices.append(i)
            dataset = dataset[torch.tensor(indices)]

            if dataset.transform is None:
                dataset.transform = T.ToDense(num_nodes)
            else:
                dataset.transform = T.Compose(
                    [dataset.transform, T.ToDense(num_nodes)])

        return dataset


def print_dataset(dataset):
    num_nodes = num_edges = 0
    for data in dataset:
        num_nodes += data.num_nodes
        num_edges += data.num_edges

    print('Name', dataset)
    print('Graphs', len(dataset))
    print('Nodes', num_nodes / len(dataset))
    print('Edges', (num_edges // 2) / len(dataset))
    print('Features', dataset.num_features)
    print('Classes', dataset.num_classes)
    print()


class gl_graph(object):
    def __init__(self, x, edge_index, batch, y, new_x, new_edge_index):
        self.x = x,
        self.edge_index = edge_index,
        self.y = y,
        self.batch = batch,
        self.new_x = new_x,
        self.new_edge_index = new_edge_index

def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask

def batch_normalize_adj(mx, mask=None):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = torch.clamp(mx.sum(1), min=VERY_SMALL_NUMBER)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    if mask is not None:
        r_inv_sqrt = r_inv_sqrt * mask

    r_mat_inv_sqrt = []
    for i in range(r_inv_sqrt.size(0)):
        r_mat_inv_sqrt.append(torch.diag(r_inv_sqrt[i]))

    r_mat_inv_sqrt = torch.stack(r_mat_inv_sqrt, 0)
    return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


def get_binarized_kneighbors_graph(features, top_k, mask=None, device=None):
    assert features.requires_grad is False
    # Compute cosine similarity matrix
    features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
    attention = torch.matmul(features_norm, features_norm.transpose(-1, -2))

    if mask is not None:
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(1), 0)
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(-1), 0)

    # Extract and Binarize kNN-graph
    top_k = min(top_k, attention.size(-1))
    _, knn_ind = torch.topk(attention, top_k, dim=-1)
    adj = torch.zeros_like(attention).scatter_(-1, knn_ind, 1).to(device)
    return adj


def create_mask(x, N, device=None):
    if isinstance(x, torch.Tensor):
        x = x.data
    mask = np.zeros((len(x), N))
    for i in range(len(x)):
        mask[i, :x[i]] = 1
    return torch.Tensor(mask).to(device)


def to_data_list(x, edge_index, y, batch):
    idx_max = batch.max().item()
    data_list = []
    base_num = 0
    for graph_id in range(idx_max+1):
        node_idx = [i for i in range(len(batch)) if batch[i] == graph_id]
        new_x = x[node_idx]
        new_edge_index = subgraph(node_idx, edge_index)[0]
        new_edge_index = new_edge_index - base_num

        data = Data(x=new_x, edge_index=new_edge_index, y=[y[graph_id]])
        data_list.append(data)
        base_num += len(node_idx)
    return data_list


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def remove_edge_random(data, remove_edge_fraction):
    """Randomly remove a certain fraction of edges."""
    data_c = deepcopy(data)
    num_edges = int(data_c.edge_index.shape[1] / 2)
    num_removed_edges = int(num_edges * remove_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data_c.edge_index.T)]
    for i in range(num_removed_edges):
        idx = np.random.choice(len(edges))
        edge = edges[idx]
        edge_r = (edge[1], edge[0])
        edges.pop(idx)
        try:
            edges.remove(edge_r)
        except:
            pass
    data_c.edge_index = torch.LongTensor(np.array(edges).T).to(data.edge_index.device)
    return data_c


def add_random_edge(data, added_edge_fraction=0):
    """Add random edges to the original data's edge_index."""
    if added_edge_fraction == 0:
        return data
    data_c = deepcopy(data)
    num_edges = int(data.edge_index.shape[1] / 2)
    num_added_edges = int(num_edges * added_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data.edge_index.T)]
    added_edges = []
    for i in range(num_added_edges):
        while True:
            added_edge_cand = tuple(np.random.choice(data.x.shape[0], size=2, replace=False))
            added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
            if added_edge_cand in edges or added_edge_cand in added_edges:
                if added_edge_cand in edges:
                    assert added_edge_r_cand in edges
                if added_edge_cand in added_edges:
                    assert added_edge_r_cand in added_edges
                continue
            else:
                added_edges.append(added_edge_cand)
                added_edges.append(added_edge_r_cand)
                break

    added_edge_index = torch.LongTensor(np.array(added_edges).T).to(data.edge_index.device)
    data_c.edge_index = torch.cat([data.edge_index, added_edge_index], 1)
    return data_c


def load_smallset():
    data_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', '20News', 'processed', 'small_set')
    small_trainset = torch.load(osp.join(data_dir, "trainset.pt"))
    small_valset = torch.load(osp.join(data_dir, "valset.pt"))
    small_testset = torch.load(osp.join(data_dir, "testset.pt"))
    return small_trainset, small_valset, small_testset


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def config_args(args):
    if args.dataset_name == "REDDIT-BINARY":
        args.epochs = 200
        args.feature_denoise = False
        if args.backbone == "GAT":
            args.batch_size = 30
            args.test_batch_size = 10
        else:
            args.batch_size = 80
            args.test_batch_size = 60
    return args
