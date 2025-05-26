import torch_geometric
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import json
import GraphDataSets
import GraphNN
import GraphTransforms as Tr
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import (
    GCNConv, GraphConv, GATConv, GATv2Conv, TransformerConv
)

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Parche para nombres de variables de entorno
cluster = os.environ.get('cluster_var', 'default_cluster')
process = os.environ.get('process_var', 'default_process')
#save_path = f'./jobs_aux/{cluster}_{process}/'
#creating output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Crear un nombre único para el directorio
#save_path = os.path.join("output", f"out_job_{timestamp}")
save_path = f"out_job_{timestamp}"
os.makedirs(save_path, exist_ok=True)

# Configuración inicial
print('Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if torch.cuda.is_available():
    print('Memoria asignada:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')

# Configuración del entrenamiento
config = {
    "dataset_name": "RecoNew_all_10mm_KNN20_hitsopt",  # Nombre del dataset a cargar
    "batch_size": 128,                      # Tamaño del batch
    "learning_rate": 1e-4,                  # Tasa de aprendizaje
    "n_epochs": 300,                        # Número máximo de épocas
    "dropout": 0.1,                         # Dropout para regularización
    "dropoutLayer": 0.4,                    # Dropout para cada layer
    "attention_heads": 4,                   # Number of Attention Heads needed for GAT,GATv2,TransformerConv ..
    "early_stop": True,                     # Activar parada temprana
    "patience": 20,                         # Número de épocas sin mejora antes de parar
    "convtype": "TransformerConv",          # supported: GCNConv, GraphConv, GATConv, GATv2Conv, TransformerConv  
    "nConvLayers": 4,                       # Number of convolutional layers
    "hidden_dim": 64,                       # Dimensiones ocultas de las capas del modelo
    "doAugmentation": False,                # incluir augmentation (rotaciones de X,Y con angulo = pi/2)
    "useGroupAddRev": False,                # operaciones invertibles reducir consumo memoria
    "transform": "RandomNodeDeletion",      # Transformación de datos
    "network_file": "out_job_20250521_130456/best_model.pth",        # Nombre del archivo de red a cargar
    "Comment": "KNN20 with random node deletion" # Comentario para el entrenamiento
}

# Guardar configuración en JSON
with open(os.path.join(save_path, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

if not os.path.exists('./Input_Dataframes/MC_dataset_Marija_10mm_all.h5'):
    raise FileNotFoundError("El archivo de dataset no existe en la ruta especificada.")
print('****************** Settings **********************')
print('Batch size:', config['batch_size'])
print('Learning rate:', config['learning_rate'])
print('Epochs:', config['n_epochs'])
print('Dropout:', config['dropout'])
print('Capa de convolucion:', config['convtype'])
print('Numero de layers:', config['nConvLayers'])
print('Dropout cada layers:', config['dropoutLayer'])
print('Hidden dimensions:', config['hidden_dim'])
print('Attention heads:', config['attention_heads'])
print('doAugmentation:', config['doAugmentation'])
print('useGroupAddRev:', config['useGroupAddRev'])
print('**************************************************  ')

# Dataset
# Carga del Dataset
DatasetClass = getattr(GraphDataSets, config["dataset_name"])
transform = False
try:
    Transform = getattr(Tr, config["transform"])
    transform = True
    dataset = DatasetClass(root='./GNN_datasets/',transform=Transform())
except AttributeError:
    if config["transform"] == "knn":
        Transform = T.Compose([Tr.RandomNodeDeletion(), T.KNNGraph(k=10), T.ToUndirected(), Tr.AddEdgeEdiff()])
        transform = True
        dataset = DatasetClass(root='./GNN_datasets/',transform=Transform)
    else:
        dataset = DatasetClass(root='./')
    
full_length = len(dataset)
print('dataset successfully loaded!')
print('number of events:', full_length)
print('Características de nodo:', dataset.num_node_features)
print('Número de clases:', dataset.num_classes)

# División del dataset
train_size = int(0.70 * len(dataset))
valid_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - valid_size
def deterministic_split(dataset, train_size, valid_size, test_size, seed=42):
    """Splits the dataset deterministically using a fixed random seed."""
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, valid_size, test_size], generator=generator)

train_dataset, valid_dataset, test_dataset = deterministic_split(dataset, train_size, valid_size, test_size)

print(f"Events in train dataset: {len(train_dataset)}")
print(f"Events in validation dataset: {len(valid_dataset)}")
print(f"Events in testing dataset: {len(test_dataset)}")

if config['doAugmentation']:

    print('Performing Augmentation:')

    from torch_geometric.data import Batch
    from GraphNN_utils import FixedRotation
    from GraphNN_utils import AugmentedDataset

    x_indices = [0, 1]
    rotation_0 = FixedRotation(angle_degrees=0, x_indices=x_indices)
    rotation_90 = FixedRotation(angle_degrees=90, x_indices=x_indices)
    rotation_180 = FixedRotation(angle_degrees=180, x_indices=x_indices)
    rotation_270 = FixedRotation(angle_degrees=270, x_indices=x_indices)
    # Lista de transformaciones
    transforms_list = [rotation_0, rotation_90, rotation_180, rotation_270]

    augmented_train_dataset = AugmentedDataset(train_dataset, transforms=transforms_list)
    augmented_valid_dataset = AugmentedDataset(valid_dataset, transforms=transforms_list)
    print(f'Events in train dataset with augmentation: {len(augmented_train_dataset)}')
    print(f'Events in validation dataset with augmentation: {len(augmented_valid_dataset)}')
 
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    train_augm_loader = DataLoader(augmented_train_dataset,batch_size=config['batch_size'],shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_augm_loader = DataLoader(augmented_valid_dataset,batch_size=config['batch_size'],shuffle=True)
    print(f'Number of batches in training: {len(train_loader)}')
    print(f'Number of batches in training incluing augmentation: {len(train_augm_loader)}')
    print(f'Number of batches in validation: {len(valid_loader)}')
    print(f'Number of batches in validation incluing augmentation: {len(valid_augm_loader)}')
    print('Augmentation successfully done!')

# DataLoaders
if config['doAugmentation']:
    train_loader = train_augm_loader
    valid_loader = valid_augm_loader
else:
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

#valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
print(f"Number of batches in training: {len(train_loader)}")
print(f"Number of batches in validation: {len(valid_loader)}")
print(f"Number of batches in testing: {len(test_loader)}")

conv_type = torch_geometric.nn.TransformerConv
if config['convtype'] == "GCNConv":
    conv_type = torch_geometric.nn.GCNConv
elif config['convtype'] == "GraphConv":
    conv_type = torch_geometric.nn.GraphConv
elif config['convtype'] == "GATConv":
    conv_type = torch_geometric.nn.GATConv
elif config['convtype'] == "GATv2Conv":
    conv_type = torch_geometric.nn.GATv2Conv
elif config['convtype'] == "TransformerConv":
    conv_type = torch_geometric.nn.TransformerConv
else:
    raise ValueError(f"Tipo de convolución desconocido: {config['convtype']}")

from GraphNN import GNN
model = GNN(
    conv_type=conv_type, ## supported: GCNConv, GraphConv, GATConv, GATv2Conv, TransformerConv
    hidden_dim=config['hidden_dim'],
    dataset=dataset,
    dropout=config['dropout'],
    dropoutLayer=config['dropoutLayer'],
    heads=config['attention_heads'],
    edge_dim=dataset.edge_attr.size(1),
    num_layers=config['nConvLayers']
    #use_groupaddrev=config['useGroupAddRev'],
    #num_groups=2
).to(device)

try:
    model.load_state_dict(torch.load(config["network_file"]))
    model.eval()
    print('Model successfully loaded!')
except:
    print('No model loaded!')

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = nn.CrossEntropyLoss(reduction='mean')

# Funciones auxiliares
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_with_predictions(model, loader, criterion, device):
    """Evaluates the model and returns loss, true labels, and predictions."""
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    for data in tqdm(loader, desc="Validating"):
        data = data.to(device)
        data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(output, data.y)
        total_loss += loss.item()
        y_true.append(data.y.cpu().numpy())
        y_pred.append(output.softmax(dim=-1).cpu().numpy())
    return total_loss / len(loader), np.concatenate(y_true), np.concatenate(y_pred)

# Guardar datos y pérdidas
store = pd.HDFStore(f'{save_path}/Loss_{config["dataset_name"]}.h5')

# Entrenamiento
best_loss = float('inf')
stale_epochs = 0
loss_train, loss_valid = [], []

for epoch in range(config['n_epochs']):
    print(f"Epoch {epoch + 1}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    valid_loss, y_valid, y_predict_valid = evaluate_with_predictions(
        model, valid_loader, criterion, device
    )

    print(f"Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")
    loss_train.append(train_loss)
    loss_valid.append(valid_loss)

    if valid_loss < best_loss:
        best_loss = valid_loss
        stale_epochs = 0
        torch.save(model.state_dict(), f"{save_path}/best_model.pth")
    else:
        stale_epochs += 1

    if config['early_stop'] and stale_epochs >= config['patience']:
        print("Early stopping triggered.")
        break

# Evaluación en conjunto de prueba
model.load_state_dict(torch.load(f"{save_path}/best_model.pth"))
test_loss, y_test, y_predict_test = evaluate_with_predictions(
    model, test_loader, criterion, device
)
train_loss, y_train, y_predict_train = evaluate_with_predictions(
    model, train_loader, criterion, device
)

# Guardar resultados en HDF5
store["loss_train"] = pd.DataFrame(loss_train, columns=["Train Loss"])
store["loss_valid"] = pd.DataFrame(loss_valid, columns=["Validation Loss"])
store["y_train"] = pd.DataFrame(y_train, columns=["True Labels"])
store["y_predict_train"] = pd.DataFrame(y_predict_train, columns=["Predicted Probability Class 0", "Predicted Probability Class 1"])
store["y_valid"] = pd.DataFrame(y_valid, columns=["True Labels"])
store["y_predict_valid"] = pd.DataFrame(y_predict_valid, columns=["Predicted Probability Class 0", "Predicted Probability Class 1"])
store["y_test"] = pd.DataFrame(y_test, columns=["True Labels"])
store["y_predict_test"] = pd.DataFrame(y_predict_test, columns=["Predicted Probability Class 0", "Predicted Probability Class 1"])
store.close()

# Guardar información del entrenamiento
traininfo_path = f'{save_path}/train_info.txt'
start_time = datetime.now()
with open(traininfo_path, 'w') as f:
    f.write(f"Training configuration:\n")
    for key, value in config.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nBest epoch: {epoch + 1}\n")
    f.write(f"Final train loss: {train_loss:.4f}\n")
    f.write(f"Final validation loss: {valid_loss:.4f}\n")
    f.write(f"Final test loss: {test_loss:.4f}\n")

# Guardar tiempo de ejecución
end_time = datetime.now()
with open(traininfo_path, 'a') as f:
    f.write(f"\nTraining started at: {start_time}\n")
    f.write(f"Training ended at:   {end_time}\n")
    f.write(f"Total training time: {end_time - start_time}\n")

print("Training and evaluation complete. Results saved.")
