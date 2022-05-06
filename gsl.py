import torch
from layers import *
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout
from torch.autograd import Variable
from torch_geometric.data import Data
from utils import *
from layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VIBGSL(torch.nn.Module):
    def __init__(self, args, num_node_features, num_classes):
        super(VIBGSL, self).__init__()
        self.args = args
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.backbone = args.backbone
        self.hidden_dim = args.hidden_dim
        self.IB_size = args.IB_size
        self.graph_metric_type = args.graph_metric_type
        self.graph_type = args.graph_type
        self.top_k = args.top_k
        self.epsilon = args.epsilon
        self.beta = args.beta
        self.num_per = args.num_per

        if self.backbone == "GCN":
            self.backbone_gnn = myGCN(self.args, in_dim=self.num_node_features, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)
        elif self.backbone == "GIN":
            self.backbone_gnn = myGIN(self.args, in_dim=self.num_node_features, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)
        elif self.backbone == "GAT":
            self.backbone_gnn = myGAT(self.args, in_dim=self.num_node_features, out_dim=self.IB_size*2,
                                      hidden_dim=self.hidden_dim)

        self.graph_learner = GraphLearner(input_size=self.num_node_features, hidden_size=self.hidden_dim,
                                          graph_type=self.graph_type, top_k=self.top_k,
                                          epsilon=self.epsilon, num_pers=self.num_per, metric_type=self.graph_metric_type,
                                          feature_denoise=self.args.feature_denoise, device=None)

        self.classifier = torch.nn.Sequential(Linear(self.IB_size, self.IB_size), ReLU(), Dropout(p=0.5),
                                              Linear(self.IB_size, self.num_classes))

        if torch.cuda.is_available():
            self.backbone_gnn = self.backbone_gnn.cuda()
            self.graph_learner = self.graph_learner.cuda()
            self.classifier = self.classifier.cuda()

    def __repr__(self):
        return self.__class__.__name__

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def to(self, device):
        self.backbone_gnn.to(device)
        self.graph_learner.to(device)
        self.classifier.to(device)
        return self

    def reset_parameters(self):
        self.backbone_gnn.reset_parameters()
        self.graph_learner.reset_parameters()
        for module in self.classifier:
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()

    def learn_graph(self, node_features, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        new_feature, new_adj = self.graph_learner(node_features)

        if graph_skip_conn in (0.0, None):
            # add I
            if graph_include_self:
                if torch.cuda.is_available():
                    new_adj = new_adj + torch.eye(new_adj.size(0)).cuda()
                else:
                    new_adj = new_adj + torch.eye(new_adj.size(0))
        else:
            # skip connection
            new_adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * new_adj

        return new_feature, new_adj

    def reparametrize_n(self, mu, std, n=1):
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + eps * std

    def forward(self, graphs):
        num_sample = graphs.num_graphs
        graphs_list = graphs.to_data_list()
        new_graphs_list = []

        for graph in graphs_list:
            x, edge_index = graph.x.to(device), graph.edge_index.to(device)
            raw_adj = to_dense_adj(edge_index)[0]
            new_feature, new_adj = self.learn_graph(node_features=x,
                                                    graph_skip_conn=self.args.graph_skip_conn,
                                                    graph_include_self=self.args.graph_include_self,
                                                    init_adj=raw_adj)
            new_edge_index, new_edge_attr = dense_to_sparse(new_adj)

            new_graph = Data(x=new_feature, edge_index=new_edge_index, edge_attr=new_edge_attr)
            new_graphs_list.append(new_graph)
        loader = DataLoader(new_graphs_list, batch_size=len(new_graphs_list))
        batch_data = next(iter(loader))
        node_embs, _ = self.backbone_gnn(batch_data.x, batch_data.edge_index)
        graph_embs = global_mean_pool(node_embs, batch_data.batch)


        mu = graph_embs[:, :self.IB_size]
        std = F.softplus(graph_embs[:, self.IB_size:]-self.IB_size, beta=1)
        new_graph_embs = self.reparametrize_n(mu, std, num_sample)

        logits = self.classifier(new_graph_embs)

        return (mu, std), logits, graphs_list, new_graphs_list


