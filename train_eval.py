import time
import math
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Dataset, Batch, DataLoader, DenseDataLoader as DenseLoader
from gsl import *
from utils import *
from param_parser import parameter_parser
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(6789)
np.random.seed(6789)
torch.cuda.manual_seed_all(6789)
os.environ['PYTHONHASHSEED'] = str(6789)


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size, test_batch_size, lr,
                                  lr_decay_factor, lr_decay_step_size, weight_decay, logger=None):
    val_losses, val_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        if isinstance(dataset, Dataset):
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            val_dataset = dataset[val_idx]
        elif isinstance(dataset, list):
            train_dataset = [dataset[idx] for idx in train_idx.numpy().tolist()]
            test_dataset = [dataset[idx] for idx in test_idx.numpy().tolist()]
            val_dataset = [dataset[idx] for idx in val_idx.numpy().tolist()]

        fold_val_losses = []
        fold_val_accs = []
        fold_test_accs = []

        infos = dict()

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, test_batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, test_batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, test_batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            if model.__repr__() in ['GCN', 'GIN', 'GAT']:
                train_loss, train_acc = train(model, optimizer, train_loader)
                val_loss, val_acc = eval_loss(model, val_loader)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                fold_val_losses.append(val_loss)
                fold_val_accs.append(val_acc)
                test_acc = eval_acc(model, test_loader)
                test_accs.append(test_acc)
                fold_test_accs.append(test_acc)
                eval_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'test_acc': test_accs[-1],
                }
            elif model.__repr__() in ['VIBGSL']:
                train_cls_loss, train_KL_loss, train_loss, train_acc = train_VGIB(model, optimizer, train_loader)
                val_cls_loss, val_KL_loss, val_loss, val_acc = eval_VGIB_loss(model, val_loader)
                
                val_losses.append(val_loss)
                fold_val_losses.append(val_loss)
                fold_val_accs.append(val_acc)
                val_accs.append(val_acc)

                test_acc, data, graphs_list, new_graphs_list, pred_y = eval_VGIB_acc(model, test_loader)
                test_accs.append(test_acc)
                fold_test_accs.append(test_acc)
                eval_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_cls_loss': val_cls_loss,
                    "val_KL_loss": val_KL_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                }
                infos[epoch] = eval_info
            else:
                raise ValueError('Unknown model: {}'.format(model.__repr__()))

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            if epoch % 10 == 0:
                print('Epoch: {:d}, train loss: {:.3f}, train acc: {:.3f}, val loss: {:.5f}, val acc: {:.3f}, test scc: {:.3f}'
                      .format(epoch, eval_info["train_loss"], eval_info["train_acc"], eval_info["val_loss"], eval_info["val_acc"], eval_info["test_acc"]))

        fold_val_loss, argmin = tensor(fold_val_losses).min(dim=0)
        fold_val_acc, argmax = tensor(fold_val_accs).max(dim=0)
        fold_test_acc = fold_test_accs[argmin]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        print('Fold: {:d}, train acc: {:.3f}, Val loss: {:.3f}, Val acc: {:.5f}, Test acc: {:.3f}'
              .format(eval_info["fold"], eval_info["train_acc"], fold_val_loss, fold_val_acc, fold_test_acc))


    val_losses, val_accs, test_accs, duration = tensor(val_losses), tensor(val_accs), tensor(test_accs), tensor(durations)
    val_losses, val_accs, test_accs = val_losses.view(folds, epochs), val_accs.view(folds, epochs), test_accs.view(folds, epochs)


    min_val_loss, argmin = val_losses.min(dim=1)
    max_val_acc, argmax = val_accs.max(dim=1)
    test_acc = test_accs[torch.arange(folds, dtype=torch.long), argmin]

    val_loss_mean = min_val_loss.mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()
    print(test_acc)
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f}+{:.3f}, Duration: {:.3f}'
          .format(val_loss_mean, test_acc_mean, test_acc_std, duration_mean))

    return test_acc, test_acc_mean, test_acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=6789)

    test_indices, train_indices = [], []
    if isinstance(dataset, Dataset):
        for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
            test_indices.append(torch.from_numpy(idx).to(torch.long))
    elif isinstance(dataset, list):
        for _, idx in skf.split(torch.zeros(len(dataset)), [data.y for data in dataset]):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()
    model.to(device)
    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)

        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def train_VGIB(model, optimizer, loader):
    model.train()

    total_loss = 0
    total_class_loss = 0
    total_KL_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        (mu, std), logits, _, _ = model(data)
        class_loss = F.cross_entropy(logits, data.y).div(math.log(2))
        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))

        loss = class_loss + model.beta * KL_loss
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_class_loss += class_loss.item() * num_graphs(data)
        total_KL_loss += KL_loss.item() * num_graphs(data)
        optimizer.step()
        pred = logits.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return total_class_loss / len(loader.dataset), total_KL_loss / len(loader.dataset), total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_VGIB_acc(model, loader):
    model.eval()

    correct = 0
    graphs_list = []
    new_graphs_list = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            _, logits, tmp_graphs_list, tmp_new_graphs_list = model(data)
            graphs_list += tmp_graphs_list
            new_graphs_list += tmp_new_graphs_list
            pred = logits.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset), data, graphs_list, new_graphs_list, pred


def eval_VGIB_loss(model, loader):
    model.eval()
    total_loss = 0
    total_class_loss = 0
    total_KL_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            (mu, std), logits, _, _ = model(data)
            pred = logits.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        class_loss = F.cross_entropy(logits, data.y).div(math.log(2))
        KL_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        loss = class_loss.item() + model.beta * KL_loss.item()
        total_loss += loss * num_graphs(data)
        total_class_loss += class_loss * num_graphs(data)
        total_KL_loss += KL_loss * num_graphs(data)
    return total_class_loss.item() / len(loader.dataset), total_KL_loss.item() / len(loader.dataset), total_loss / len(loader.dataset), correct / len(loader.dataset)


def testacc():
    args = parameter_parser()
    if args.dataset_name in ["IMDB-BINARY", "REDDIT-BINARY", "COLLAB", "IMDB-MULTI"]:
        dataset = get_dataset(args.dataset_name)
    test_batch_size = args.test_batch_size
    folds = args.folds
    modelname = args.dataset_name+"_"+args.backbone
    _, test_idx, _ = k_fold(dataset, folds)
    if isinstance(dataset, Dataset):
        test_dataset = dataset[test_idx[0]]
    elif isinstance(dataset, list):
        test_dataset = [dataset[idx] for idx in test_idx[0].numpy().tolist()]
    if 'adj' in test_dataset[0]:
        test_loader = DenseLoader(test_dataset, test_batch_size, shuffle=False)
    else:
        test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False)
    model = torch.load("results/"+modelname+".pth")
    test_acc, data, graphs_list, new_graphs_list, pred_y = eval_VGIB_acc(model, test_loader)
    print("Test acc of "+modelname+" is: "+str(test_acc))


if __name__ == '__main__':
    testacc()