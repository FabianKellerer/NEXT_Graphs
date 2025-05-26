import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import GraphTransforms as Tr
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import GraphNN as GD
import GraphDataSets as G
from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split
import os.path as osp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#import mplhep as hep
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

from datetime import datetime

import os

print('Packages importing successful.')

cluster = os.environ.get('cluster_var')
process = os.environ.get('process_var')
home_path = './jobs_aux/'
save_path = f'{home_path}{cluster}_{process}/'
#os.system(f'mkdir -p {save_path}')
traininfo_path = f'{save_path}train_info.txt'
with open(traininfo_path, 'w') as f:
    f.write(f'cluster: {cluster}\n')
    f.write(f'process: {process}')


print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

start_time = datetime.now() #.strftime('%Y/%m/%d %H:%M:%S')

#############################
#          SETTINGS         #
#############################

# Graph dataset
dataset_name = 'RecoNew_all_10mm_R2_voxopt_Paolina'
transform = False
# Layers cell size
hidden  = 64
# Output layer dropout
batch_size = 128
learning_rate = 1e-3
n_epochs = 400
dropout = 0.1
early_stop = True
patience = 20

#############################


# if radius==1.1:
#     dataset   = GD.Truth_SB50_5mm_R1(root='/lhome/ific/a/antalo/TFG_NEXT/workarea/GNN_datasets/R1/')
# elif radius==2.1:
#     dataset   = GD.Truth_SB50_5mm_R2(root='/lhome/ific/a/antalo/TFG_NEXT/workarea/GNN_datasets/R2/')

DS      = getattr(G,dataset_name)
if transform:
    dataset = DS(root='./GNN_datasets/',transform=Tr.RandomNodeSplit())#, pre_transform=transform)
    dataset_name += '_T'
else:
    dataset = DS(root='./GNN_datasets/')

print('Graph dataset building/importing successful.')

graph_dataset = dataset[:int(2/3*len(dataset))]
test_dataset  = dataset[int(2/3*len(dataset)):]

inputs  = dataset.num_node_features
# # Layers cell size
# hidden  = 64
outputs = dataset.num_classes
# # Output layer dropout
# dropout = 0.5

model = GD.GCN(hidden,dataset=dataset, dropout=dropout).to(device)
# learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

@torch.no_grad()
def test(model, loader, total, batch_size, leave=False):
    model.eval()
    
    xentropy = nn.CrossEntropyLoss(reduction='mean')

    sum_loss = 0.
    t = tqdm(enumerate(loader), total=total/batch_size, leave=leave)
    for i, data in t:
        data = data.to(device)
        y = data.y
        batch_output = model(data.x, data.edge_index, data.batch)
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
        batch_output = model(data.x, data.edge_index, data.batch)
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

torch.manual_seed(0)
valid_frac = 0.20
full_length = len(graph_dataset)
valid_num = int(valid_frac*full_length)
# batch_size = 250

train_dataset, valid_dataset = random_split(graph_dataset, [full_length-valid_num,valid_num])



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_samples = len(train_dataset)
valid_samples = len(valid_dataset)
test_samples = len(test_dataset)
print(full_length)
print(train_samples)
print(valid_samples)
print(test_samples)

# n_epochs = 150
# n_epochs = 2
stale_epochs = 0
best_valid_loss = 99999
# patience = 10
t = tqdm(range(0, n_epochs))
loss_train = []
loss_valid = []

for epoch in t:
    loss = train(model, optimizer, train_loader, train_samples, batch_size, leave=bool(epoch==n_epochs-1))
    valid_loss = test(model, valid_loader, valid_samples, batch_size, leave=bool(epoch==n_epochs-1))
    print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
    print('           Validation Loss: {:.4f}'.format(valid_loss))
    loss_train.append(loss)
    loss_valid.append(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        modpath = osp.join(f'GraphConv_{dataset_name}_best.pth')
        print('New best model saved to:',modpath)
        torch.save(model.state_dict(),modpath)
        stale_epochs = 0
        best_epoch = epoch + 1
    else:
        print('Stale epoch')
        stale_epochs += 1

    if early_stop==True and stale_epochs >= patience:
        print('Early stopping after %i stale epochs'%patience)
        break

print('Training successful.')

model.eval()

t_train = tqdm(enumerate(train_loader),total=train_samples/batch_size)
t_valid = tqdm(enumerate(valid_loader),total=valid_samples/batch_size)
t_test = tqdm(enumerate(test_loader),total=test_samples/batch_size)

y_train = []
y_predict_train = []
y_valid = []
y_predict_valid = []
y_test = []
y_predict_test = [] 

for i,data in t_train:
    data = data.to(device)
    batch_output = model(data.x, data.edge_index, data.batch)
    for j,obj in enumerate(batch_output.detach().cpu().numpy()):
        y_predict_train.append(batch_output.detach().cpu().numpy()[j][1])
    y_train.append(data.y.cpu().numpy())
y_train = np.concatenate(y_train)
y_predict_train = np.array(y_predict_train)

for i,data in t_valid:
    data = data.to(device)
    batch_output = model(data.x, data.edge_index, data.batch)
    for j,obj in enumerate(batch_output.detach().cpu().numpy()):
        y_predict_valid.append(batch_output.detach().cpu().numpy()[j][1])
    y_valid.append(data.y.cpu().numpy())
y_valid = np.concatenate(y_valid)
y_predict_valid = np.array(y_predict_valid)

for i,data in t_test:
    data = data.to(device)
    batch_output = model(data.x, data.edge_index, data.batch)
    for j,obj in enumerate(batch_output.detach().cpu().numpy()):
        y_predict_test.append(batch_output.detach().cpu().numpy()[j][1])
    y_test.append(data.y.cpu().numpy())
y_test = np.concatenate(y_test)
y_predict_test = np.array(y_predict_test)

end_time = datetime.now() #.strftime('%Y/%m/%d %H:%M:%S')

print('Evaluation successful.')

store=pd.HDFStore(f'Loss_GCN_{dataset_name}.h5')
store["loss_train"]   = pd.DataFrame(loss_train, columns = ['Epoch'])
store["loss_valid"]   = pd.DataFrame(loss_valid, columns = ['Epoch'])
store["y_train"]    = pd.DataFrame(y_train, columns = ['Event'])
store["y_predict_train"] = pd.DataFrame(y_predict_train, columns = ['Event'])
store["y_valid"]    = pd.DataFrame(y_valid, columns = ['Event'])
store["y_predict_valid"] = pd.DataFrame(y_predict_valid, columns = ['Event'])
store["y_test"]    = pd.DataFrame(y_test, columns = ['Event'])
store["y_predict_test"] = pd.DataFrame(y_predict_test, columns = ['Event'])
store.close()

with open(traininfo_path, 'a') as f:
    f.write(f'\n\n')
    f.write(f'max epochs: {n_epochs}\n')
    f.write(f'early-stopping applied: {early_stop}\n')
    f.write(f'best epoch: {best_epoch}\n')
    f.write(f'patience: {patience}\n')
    f.write(f'batch size: {batch_size}\n')
    f.write(f'learning_rate: {learning_rate}\n')
    f.write(f'layer cell size: {hidden}\n')
    f.write(f'output layer dropout: {dropout}')

with open(traininfo_path, 'a') as f:
    f.write(f'\n\n')
    f.write(f'Training start: {start_time}\n')
    f.write(f'Training end:   {end_time}\n')
    f.write(f'Training time:  {end_time - start_time}')
