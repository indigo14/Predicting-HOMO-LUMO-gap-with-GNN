import logging
from typing import Optional
import pandas as pd
import rdkit
import dgl
import dgllife
from rdkit import Chem
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
mol_to_bigraph
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor
from sklearn.metrics import r2_score


class Runner:
    def __init__(
        self,
        # train_loader: GeoDataLoader,
        # val_loader: GeoDataLoader,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: Optional[Optimizer] = None,
        epochs: int = 5
    ) -> None:
        """
        Initializes the Runner.

        Parameters:
        loader (GeoDataLoader): The data loader for fetching batches of data.
        model (torch.nn.Module): The neural network model to train or evaluate.
        device (torch_device): The device (CPU or GPU) to perform computations on.
        optimizer (Optional[Optimizer]): The optimizer for training the model. If None, the runner is in evaluation mode.
        epochs (int): Number of epochs to train the model.
        """


        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.score_train = []
        self.score_val = []
        self.is_train = optimizer is not None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Runner")
        self._prepare_data()

    def _prepare_data(self):
        self.logger.info("Preparing data")
        self.logger.info("Data preparation complete")

    def train(self):
        for epoch in range(self.epochs):
            self.logger.info(f"Starting Epoch {epoch + 1}")
            self.model.train()
            train_loss = []
            all_train_targets = []
            all_train_predictions = []

            for batch in self.train_loader:
                batch_graph, target = batch
                node_feats = batch_graph.ndata["hv"]
                edge_feats = batch_graph.edata["he"]

                batch_graph = batch_graph.to(self.device)
                edge_feats = edge_feats.to(self.device)
                node_feats = node_feats.to(self.device)
                target = target.to(self.device)

                predictions = self.model(batch_graph, node_feats, edge_feats)
                predictions = predictions.view(-1)

                loss = torch.nn.MSELoss()(predictions, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                all_train_targets.extend(target.cpu().detach().numpy())
                all_train_predictions.extend(predictions.cpu().detach().numpy())

            avg_train_loss = torch.tensor(train_loss).mean().item()
            train_r2 = r2_score(all_train_targets, all_train_predictions)
            self.logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}, R2: {train_r2:.4f}")
            self.score_train.append(train_r2)

            val_loss, val_r2 = self.validate()
            self.logger.info(f"Epoch {epoch + 1} - Validation loss: {val_loss:.4f}, R2: {val_r2:.4f}")
            self.score_val.append(val_r2)

        return self.score_train, self.score_val

    def validate(self):
        self.model.eval()
        val_loss = []
        all_val_targets = []
        all_val_predictions = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch_graph, target = batch
                node_feats = batch_graph.ndata["hv"]
                edge_feats = batch_graph.edata["he"]

                batch_graph = batch_graph.to(self.device)
                edge_feats = edge_feats.to(self.device)
                node_feats = node_feats.to(self.device)
                target = target.to(self.device)

                predictions = self.model(batch_graph, node_feats, edge_feats)
                predictions = predictions.view(-1)

                loss = torch.nn.MSELoss()(predictions, target)
                val_loss.append(loss.item())

                all_val_targets.extend(target.cpu().detach().numpy())
                all_val_predictions.extend(predictions.cpu().detach().numpy())

        avg_val_loss = torch.tensor(val_loss).mean().item()
        val_r2 = r2_score(all_val_targets, all_val_predictions)
        return avg_val_loss, val_r2

    def test(self):
        self.model.eval()
        test_loss = []
        all_test_targets = []
        all_test_predictions = []

        with torch.no_grad():
            for batch in self.test_loader:
                batch_graph, target = batch
                node_feats = batch_graph.ndata["hv"]
                edge_feats = batch_graph.edata["he"]

                batch_graph = batch_graph.to(self.device)
                edge_feats = edge_feats.to(self.device)
                node_feats = node_feats.to(self.device)
                target = target.to(self.device)

                predictions = self.model(batch_graph, node_feats, edge_feats)
                predictions = predictions.view(-1)

                loss = torch.nn.MSELoss()(predictions, target)
                test_loss.append(loss.item())

                all_test_targets.extend(target.cpu().detach().numpy())
                all_test_predictions.extend(predictions.cpu().detach().numpy())

        avg_test_loss = torch.tensor(test_loss).mean().item()
        test_r2 = r2_score(all_test_targets, all_test_predictions)
        return avg_test_loss, test_r2