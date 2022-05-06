import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN


class GCN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(GCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
            x = F.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr))
            x = F.relu(self.conv2(x=x, edge_index=edge_index, edge_weight=edge_attr))
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class myGCN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        node_embeddings = x

        return node_embeddings, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(GIN, self).__init__()
        self.args = args
        self.conv1 = GINConv(
            Sequential(
                Linear(in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(self.args.num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        BN(hidden_dim),
                    ), train_eps=True))
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class myGIN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGIN, self).__init__()
        self.args = args
        self.conv1 = GINConv(
            Sequential(
                Linear(in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=True)
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, out_dim),
                ReLU(),
                BN(out_dim),
            ), train_eps=True)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        node_embeddings = x

        return node_embeddings, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__




class GAT(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(GAT, self).__init__()
        self.args = args
        self.conv1 = GATConv(in_dim, hidden_dim, heads=8, dropout=0.5)

        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1,
                             concat=False, dropout=0.5)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
            x = F.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr))
            x = F.relu(self.conv2(x=x, edge_index=edge_index, edge_weight=edge_attr))
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class myGAT(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGAT, self).__init__()
        self.args = args
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, dropout=0.5)

        self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1,
                             concat=False, dropout=0.5)
        self.relu = torch.nn.LeakyReLU(0.2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        node_embeddings = x
        return node_embeddings, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__