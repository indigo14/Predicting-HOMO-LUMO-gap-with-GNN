import requests
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, Set2Set
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, Data

from dgl.nn.pytorch import Set2Set
from dgllife.model.gnn import MPNNGNN

'''There are 3 parts to the model.  weights for the model '''