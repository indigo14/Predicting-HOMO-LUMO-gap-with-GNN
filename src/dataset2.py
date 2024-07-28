import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
mol_to_bigraph
import torch
from torch.utils.data import DataLoader, Dataset

'''Takes in a DataFrame dset with columns ["smiles"] and ["gap"]. Creates atom and bond featurizer objects. Converts SMILES to graphs. Splits the dataset into train, validation, and test sets with a ratio of 0.8:0.1:0.1.'''
class MoleculeDataset:
    def __init__(self, dset):
        self.dset = dset
        self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="hv")
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="he")
        self.dset["graph"] = self.dset["smiles"].apply(self.smiles_to_graph)
        self.train_set, self.val_set, self.test_set = self.split_dataset()

    #The method smiles_to_graph converts SMILES strings to graph objects using RDKit and DGL.
    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_bigraph(mol, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        return graph

    #The method get_data_splits returns the training, validation, and test sets.
    def split_dataset(self):
        train, temp = train_test_split(self.dset, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, val, test

    #The method get_data_splits returns the training, validation, and test sets.
    def get_data_splits(self):         
        return self.train_set, self.val_set, self.test_set

'''This class inherits from torch.utils.data.Dataset and is used to handle the DataFrame.
__len__ returns the number of samples.
__getitem__ retrieves a sample at a given index, including the graph and target value.'''        

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

'''Creates an instance of GraphDataset with the input DataFrame. The function returns a DataLoader object that can be used to iterate over the dataset in batches.'''

def create_dataloaders(dataframe, batch_size=32, shuffle=True):
    dataset = GraphDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_data)
    return dataloader          