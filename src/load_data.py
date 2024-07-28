
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
mol_to_bigraph
from fast_ml.model_development import train_valid_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import dgl

class MoleculeDataset:
    def __init__(self, dset):
        self.dset = dset
        self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="hv")
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="he")
        self.dset["graph"] = self.dset["smiles"].apply(self.smiles_to_graph)
        self.train_set, self.val_set, self.test_set = self.split_dataset()

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_bigraph(mol, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        return graph

    def split_dataset(self):
        train, temp = train_test_split(self.dset, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, val, test

    def get_data_splits(self):
      
        return self.train_set, self.val_set, self.test_set

class GraphDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        graph = row['graph']
        target = torch.tensor(row['gap'], dtype=torch.float32)
        return graph, target

def collate_data(data):
    # Unpacks the input data into separate lists of graphs and target values
    graphs, targets = map(list, zip(*data))
    
    # Batches the list of graphs into a single batched graph
    batch_graph = dgl.batch(graphs)
    
    # Stacks the list of target values into a single tensor
    targets = torch.stack(targets, dim=0)
    
    return batch_graph, targets

def create_dataloaders(dataframe, batch_size=32, shuffle=True):
    dataset = GraphDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_data)
    return dataloader        

# # create the atom and bond featurizer object
# atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="hv")
# bond_featurizer = CanonicalBondFeaturizer(bond_data_field="he")


# # helper function to convert smiles to graph
# def smiles2graph(smiles):
#   mol = Chem.MolFromSmiles(smiles)
#   graph = mol_to_bigraph(mol, node_featurizer=atom_featurizer, 
#                      edge_featurizer=bond_featurizer)
#   return graph

# # add graphs to dataframe
# dataset["graph"] = dataset["smiles"].apply(smiles2graph)


# # import the function to split into train-valid-test
# from fast_ml.model_development import train_valid_test_split

# X_train, y_train, X_valid, y_valid, \
# X_test, y_test = train_valid_test_split(dataset[["graph","gap"]], 
#                                         target = "gap", 
#                                         train_size=0.8,
#                                         valid_size=0.1, 
#                                         test_size=0.1) 

# # creating dataloader

#

# def collate_data(data):
#   # our data is in the form of list of (X,y)
#   # the map function thus maps accordingly
#   graphs, y = map(list, zip(*data))

#   # for creating a batch of graph, we use the batch function
#   batch_graph = dgl.batch(graphs)

#   # we need to stack the ys for different entries in the batch
#   y = torch.stack(y, dim=0)

#   return batch_graph, y


# import dataloader
#

# create the dataloader for train dataset
# dataset should be of form (X,y) according to the collate function
# the ys should also be converted to tensors
# train_dataloader = DataLoader(
#     dataset=list(zip(X_train["graph"].values.tolist(),
#                      torch.tensor(y_train.tolist(), dtype=torch.float32))),
#     batch_size=64, collate_fn=collate_data)

# valid_dataloader = DataLoader(
#     dataset=list(zip(X_valid["graph"].values.tolist(),
#                      torch.tensor(y_valid.tolist(), dtype=torch.float32))),
#     batch_size=64, collate_fn=collate_data)

# test_dataloader = DataLoader(
#     dataset=list(zip(X_test["graph"].values.tolist(),
#                      torch.tensor(y_test.tolist(), dtype=torch.float32))),
#     batch_size=64, collate_fn=collate_data)


# import MLP model from dgl-lifesci
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor

# the atom feature length is 74 and bond is 12
model = MPNNPredictor(node_in_feats = 74, 
                      edge_in_feats = 12, 
                      node_out_feats = 64, 
                      edge_hidden_feats = 128,
                      n_tasks = 1,
                      num_step_message_passing = 6,
                      num_step_set2set = 6,
                      num_layer_set2set = 3)


# loss function for regresssion is usually mean squared error
import torch

loss_func = torch.nn.MSELoss(reduce=None)


# adam optimier
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)