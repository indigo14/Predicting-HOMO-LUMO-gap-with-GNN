import pandas as pd
import logging
from src.dataset2 import MoleculeDataset, GraphDataset, collate_data, create_dataloaders
from src.runner5 import Runner
from src.tune2 import objective
import torch
#from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from torch.optim import Adam
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("program.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path to your dataset
DATASET_PATH = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

# Read the CSV into a DataFrame
df_init = pd.read_csv(DATASET_PATH)

'''Choose what fraction of the dataset to use for timesaving vs accuracy'''
dsfrac = 0.05 # change this to 1 if you want to use the whole dataset
dset = df_init[["smiles","gap"]].sample(frac=dsfrac)

# Ensure df_init is correctly populated
print(dset.head())
print(type(dset))

# Create the dataset object
dataset = MoleculeDataset(dset)

# Get the splits datasets still in the form of [gap], [graph]
train_set, val_set, test_set = dataset.get_data_splits()
print(f'train_set: {len(train_set)} |val_set: {len(val_set)} | Test_set: {len(test_set)}')

print(train_set.head())

# Create the dataloaders
train_loader = create_dataloaders(train_set, batch_size=32, shuffle=True)
val_loader = create_dataloaders(val_set, batch_size=32, shuffle=False)
test_loader = create_dataloaders(test_set, batch_size=32, shuffle=False)
print(f'train_loader: {len(train_loader)} |val_loader: {len(val_loader)} | Test_loader: {len(test_loader)}')
#print("train1",train_loader[1])

# Define model parameters
model_params = {
    'node_in_feats' : 74, 
    'edge_in_feats' : 12, 
    'node_out_feats' : 64, 
    'edge_hidden_feats' : 128,
    'n_tasks' : 1,
    'num_step_message_passing' : 6,
    'num_step_set2set' : 6,
    'num_layer_set2set' : 3
    }

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNNPredictor(**model_params).to(device)
# lr = best_params['lr']
# weight_decay = best_params['weight_decay']
#lr = 0.001
weight_decay = 5.259e-05  
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  
loss_func = torch.nn.MSELoss(reduce=None)


# Create the runners
train_runner = Runner(train_loader,val_loader, model, device, optimizer)
# val_runner = Runner(val_loader, model, device)
# test_runner = Runner(test_loader, model, device)
train_runner.train()
# # Training loop
# train_runner.fit(train_loader, test_loader, epochs=75)

# # Evaluate the model
# results = eval(test_loader, model, device)
# print(results)
# results = eval(train_loader, model, device)
# print(results)