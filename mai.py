import requests
import torch
import torch.nn as nn
import json

from dgl.nn.pytorch import Set2Set
from dgllife.model.gnn import MPNNGNN



class MPNN_readout(nn.Module):

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 dropout=0,
                 num_layer_set2set=3, descriptor_feats=0):
        super(MPNN_readout, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2 * node_out_feats + descriptor_feats, node_out_feats),
            nn.ReLU(),
            nn.BatchNorm1d(node_out_feats),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, concat_feats=None):
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        if concat_feats != None:
            final_feats = torch.cat((graph_feats, concat_feats), dim=1)
        else:
            final_feats = graph_feats
        return self.predict(final_feats)

with open(".\params.json") as f:
  params = json.load(f)

model = MPNN_readout(**params)
print(model)

model.load_state_dict(torch.load(".\\best_r2.pt", map_location=torch.device('cpu')))

# import from rdkit and dgl-lifesci
from rdkit import Chem
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
mol_to_bigraph

# create the atom and bond featurizer object
atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="hv")
bond_featurizer = CanonicalBondFeaturizer(bond_data_field="he")

# example smiles - ethane
smiles = "CC"

# mol_to_graph requires the RDKit molecule and featurizers
mol = Chem.MolFromSmiles(smiles)
graph = mol_to_bigraph(mol, node_featurizer=atom_featurizer,
                     edge_featurizer=bond_featurizer)

# display the graph object
print(graph)

model.eval()
node_feats = graph.ndata["hv"]
edge_feats = graph.edata["he"]
print(model(graph, node_feats, edge_feats))

torch.save(model.state_dict(), "my_model.pt")

# Load the .pt file
#file_path = "best_r2.pt"
file_path = "my_model.pt"
loaded_data = torch.load(file_path)

print(model)
for param in model.parameters():
  print(param)


for param in model.parameters():
  param.requires_grad = False

# for param in model.parameters():
#   print(param)

for param in model.parameters():
    print("Predict Params Here", param)








# # Inspect the contents
# if isinstance(loaded_data, dict):
#     # If it's a state dictionary, print the keys
#     print("State dictionary keys:")
#     for key in loaded_data.keys():
#         print(key)
# else:
#     # If it's a full model, you can inspect its structure
#     print("Loaded model:")
#     print(loaded_data)

# loaded_data = torch.load(file_path)


