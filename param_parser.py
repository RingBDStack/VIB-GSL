import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="IMDB-BINARY", help="Folder with training graph jsons.")

    parser.add_argument("--backbone", type=str, default="GCN", help="GCN, GAT, GIN")

    parser.add_argument("--graph_type", type=str, default="prob", help="epsilonNN, KNN, prob")

    parser.add_argument("--graph_metric_type", type=str, default="mlp")

    parser.add_argument("--repar", type=bool, default=True, help="Default is True.")

    parser.add_argument("--num_layers", type=int, default=2, help="Default is 2.")

    parser.add_argument("--hidden_dim", type=int, default=16, help="Default is 16.")

    parser.add_argument("--folds", type=int, default=10, help="Default is 10.")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs. Default is 200.")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate. Default is 0.001.")

    parser.add_argument('--lr_decay_factor', type=float, default=0.5)

    parser.add_argument('--lr_decay_step_size', type=int, default=50)

    parser.add_argument("--weight-decay", type=float, default=5*10**-5, help="Adam weight decay. Default is 5*10^-5.")

    parser.add_argument("--batch_size", type=int, default=100, help="batch_size")

    parser.add_argument("--test_batch_size", type=int, default=100, help="batch_size")

    parser.add_argument("--beta", type=float, default=0.00001, help="Default is 1e-5")

    parser.add_argument("--IB_size", type=int, default=16, help="Default is 16.")

    parser.add_argument("--num_per", type=int, default=16, help="Default is 16")

    parser.add_argument("--feature_denoise", type=bool, default=False, help="Default is False.")

    parser.add_argument("--top_k", type=int, default=10, help="Default is 10.")

    parser.add_argument("--epsilon", type=float, default=0.3, help="Default is 0.3.")

    parser.add_argument("--graph_skip_conn", type=float, default=0.0, help="Default is 0.0.")

    parser.add_argument("--graph_include_self", type=bool, default=True, help="Default is True.")

    return parser.parse_args()
