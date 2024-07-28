import pandas as pd
import rdkit
import dgl
import dgllife
from rdkit import Chem
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
mol_to_bigraph
import torch
from torch.utils.data import DataLoader
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor

from sklearn.metrics import r2_score

def simple_runner(loader, model, device, optimizer):
    epochs = 20
    model.to(device="cuda:0")
    
    for epoch in range(epochs):
        print("\nStarting Epoch", epoch + 1)
        model.train()
        train_loss = []
        all_targets = []
        all_predictions = []
        
        for batch in loader:
            batch_graph, target = batch

            node_feats = batch_graph.ndata["hv"]
            edge_feats = batch_graph.edata["he"]

            batch_graph = batch_graph.to(device)
            edge_feats = edge_feats.to(device)
            node_feats = node_feats.to(device)
            target = target.to(device)

            predictions = model(batch_graph, node_feats, edge_feats)
            
            # Ensure the predictions and target have the same shape
            predictions = predictions.view(-1)  # Flatten predictions to match the target shape
            
            loss = torch.nn.MSELoss()(predictions, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            # Collect all targets and predictions for R2 calculation
            all_targets.extend(target.cpu().detach().numpy())
            all_predictions.extend(predictions.cpu().detach().numpy())

        avg_loss = torch.tensor(train_loss).mean().item()
        r2 = r2_score(all_targets, all_predictions)
        
        print(f"Epoch {epoch + 1} - Training loss: {avg_loss:.4f}, R2: {r2:.4f}")





# def simple_runner(loader, model, device, loss_func, optimizer):
#     epochs = 5
#     model.to(device="cuda:0")
#     # loop over epochs
#     for epoch in range(epochs):
#         print("\nStarting Epoch", epoch+1)

#         # set the model to train so the parameters can be updated
#         model.train()

#         # loop over training batches
#         train_loss = []
#         for batch in loader: 

#             # Do a forward pass
#             batch_graph, target = batch

#             # look at the forward function for input
#             # this model needs graph, node_feats and edge_feats
#             node_feats = batch_graph.ndata["hv"]
#             edge_feats = batch_graph.edata["he"]

#             batch_graph = batch_graph.to(device)
#             edge_feats = edge_feats.to(device)
#             node_feats = node_feats.to(device)
#             target = target.to(device)

#             predictions = model(batch_graph, node_feats, edge_feats)
#             # Ensure the predictions and target have the same shape
#             predictions = predictions.view(-1)  # Flatten predictions to match the target shape
            
#             # Compute loss
#             loss = (loss_func(predictions, target)).mean()
#             optimizer.zero_grad()

#             # Do back propogation and update gradient
#             loss.backward()
#             optimizer.step()

#             # save loss to compute average loss
#             train_loss.append(loss)

#         print("Training loss", torch.tensor(train_loss).mean().item())
