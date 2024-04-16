import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import GraphNN as G
import GraphDataSets as D
from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split
import os.path as osp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import mplhep as hep
import argparse

parser = argparse.ArgumentParser('Runs the interaction graph neural network')
parser.add_argument('-d', '--dataset', required=True, type=str, help='Dataset to be used for training (Defined in the GraphNN module).')
parser.add_argument('-n', '--n_epochs', type=int, default=300, help='Number of training epochs (if early stopping condition is not met')
parser.add_argument('-hl', '--hidden_layers', type=int, default=128, help='Number of hidden layers (same for edge, node and global blocks)')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-2, help='Learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
args = parser.parse_args()


@torch.no_grad()
def test(model, loader, total, batch_size, leave=False):
    model.eval()
    
    xentropy = nn.CrossEntropyLoss(reduction='mean')

    sum_loss = 0.
    t = tqdm(enumerate(loader), total=total/batch_size, leave=leave)
    for i, data in t:
        data = data.to(device)
        y = data.y
        try:
            batch_output = model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
        except:
            batch_output = model(data.x, data.edge_index, data.edge_attr, None, data.batch)
        batch_loss_item = xentropy(batch_output, y).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)

def train(model, optimizer, loader, total, batch_size, leave=False):
    model.train()
    
    xentropy = nn.CrossEntropyLoss(reduction='mean')

    sum_loss = 0.
    t = tqdm(enumerate(loader), total=total/batch_size, leave=leave)
    for i, data in t:
        data = data.to(device)
        y = data.y
        optimizer.zero_grad()
        try:
            batch_output = model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
        except:
            batch_output = model(data.x, data.edge_index, data.edge_attr, None, data.batch)
        batch_loss = xentropy(batch_output, y)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()
    
    return sum_loss/(i+1)

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)


def Run_IN(dataset_name, n_epochs, hidden, LR, batch_size):
    
    DS = getattr(D,dataset_name)
    #transform = T.Compose([T.RadiusGraph(r=1.1), T.NormalizeScale()])
    dataset   = DS(root='./GNN_datasets/')#, pre_transform=transform)
    
    graph_dataset = dataset[:int(2/3*len(dataset))]
    test_dataset  = dataset[int(2/3*len(dataset)):]

    inputs = dataset.num_node_features
    outputs = 2

    try:
        dataset[0].edge_attr
    except AttributeError:
        dataset.edge_attr = None

    try:
        dataset[0].u
    except AttributeError:
        dataset.u = None


    model = G.InteractionNetwork(hidden, dataset=dataset, inputs=inputs, outputs=outputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    torch.manual_seed(0)
    valid_frac = 0.20
    full_length = len(graph_dataset)
    valid_num = int(valid_frac*full_length)

    train_dataset, valid_dataset = random_split(graph_dataset, [full_length-valid_num,valid_num])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples  = len(test_dataset)
    print(full_length)
    print(train_samples)
    print(valid_samples)
    print(test_samples)

    stale_epochs = 0
    best_valid_loss = 99999
    patience = 20
    t = tqdm(range(0, n_epochs))
    loss_tr = []
    loss_te = []

    for epoch in t:
        loss = train(model, optimizer, train_loader, train_samples, batch_size, leave=bool(epoch==n_epochs-1))
        valid_loss = test(model, valid_loader, valid_samples, batch_size, leave=bool(epoch==n_epochs-1))
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('           Validation Loss: {:.4f}'.format(valid_loss))
        loss_tr.append(loss)
        loss_te.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            modpath = osp.join(dataset_name + '_best.pth')
            print('New best model saved to:',modpath)
            torch.save(model.state_dict(),modpath)
            stale_epochs = 0
        else:
            print('Stale epoch')
            stale_epochs += 1
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break

    model.eval()
    t = tqdm(enumerate(test_loader),total=test_samples/batch_size)
    y_test = []
    y_predict = []
    for i,data in t:
        data = data.to(device)    
        try:
            batch_output = model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
        except:
            batch_output = model(data.x, data.edge_index, data.edge_attr, None, data.batch)
        for j,obj in enumerate(batch_output.detach().cpu().numpy()):
            y_predict.append(batch_output.detach().cpu().numpy()[j][1])
        y_test.append(data.y.cpu().numpy())
    y_test = np.concatenate(y_test)
    y_predict = np.array(y_predict)


    store=pd.HDFStore('Loss_' + dataset_name + '.h5')
    store["loss_tr"]   = pd.DataFrame(loss_tr,   columns = ['Epoch'])
    store["loss_te"]   = pd.DataFrame(loss_te,   columns = ['Epoch'])
    store.close()
    
if __name__ == '__main__':
    Run_IN(args.dataset, args.n_epochs, args.hidden_layers, args.learning_rate, args.batch_size)
