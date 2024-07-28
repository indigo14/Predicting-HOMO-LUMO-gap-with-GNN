import os
import logging
from typing import Optional
import matplotlib.pyplot as plt
import torch
import dgl
import torch.nn.functional as F
from torch_geometric.data import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import device as torch_device

class Runner:
    def __init__(
        self,
        loader: GeoDataLoader,
        model: torch.nn.Module,
        device: torch_device,
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
        self.loader = loader
        self.epochs = epochs
        self.score_train = []
        self.score_test = []



        # Assume training stage based on presence of optimizer
        self.is_train = optimizer is not None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Runner")

        self._prepare_data()

    def _prepare_data(self):
        self.logger.info("Preparing data")
        # Your data preparation logic here
        self.logger.info("Data preparation complete")

    def train(self):
        for epoch in range(self.epochs):
            self.logger.info(f"Starting Epoch {epoch + 1}")
            self.model.train()
            total_loss = 0
            total_samples = 0

            train_loss = []

            for batch in train_dataloader: 

                # Do a forward pass
                batch_graph, target = batch

                # look at the forward function for input
                # this model needs graph, node_feats and edge_feats
                node_feats = batch_graph.ndata["hv"]
                edge_feats = batch_graph.edata["he"]

                ############# transfer to GPU #################
                batch_graph = batch_graph.to(device="cuda:0")
                edge_feats = edge_feats.to(device="cuda:0")
                node_feats = node_feats.to(device="cuda:0")
                target = target.to(device="cuda:0")
                ##############################################

                predictions = model(batch_graph, node_feats, edge_feats)
            
                # Compute loss
                loss = (loss_func(predictions, target)).mean()
                optimizer.zero_grad()

                # Do back propogation and update gradient
                loss.backward()
                optimizer.step()

                # save loss to compute average loss
                train_loss.append(loss)

        print("Training loss", torch.tensor(train_loss).mean().item())

        #     for batch in self.loader:
        #         batch_graph, target = batch

        #         node_feats = batch_graph.ndata["hv"]
        #         edge_feats = batch_graph.edata["he"]

        #         batch_graph = batch_graph.to(self.device)
        #         edge_feats = edge_feats.to(self.device)
        #         node_feats = node_feats.to(self.device)
        #         target = target.to(self.device)

        #         predictions = self.model(batch_graph, node_feats, edge_feats)

        #         loss = F.mse_loss(predictions, target).mean()
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        #         train_loss.append(loss.item())
        #         total_loss += loss.item() * batch_graph.batch_size
        #         total_samples += batch_graph.batch_size

        #     avg_loss = total_loss / total_samples
        #     rmse = (avg_loss ** 0.5)
        #     self.logger.info(f"Epoch {epoch + 1} Training RMSE: {rmse:.4f}")
        #     self.score_train.append(rmse)

        # return self.score_train


# class Runner:
#     def __init__(
#         self,
#         loader: GeoDataLoader,
#         model: torch.nn.Module,
#         device: torch_device,
#         optimizer: Optional[Optimizer] = None
#     ) -> None:
#         """
#         Initializes the Runner.

#         Parameters:
#         loader (GeoDataLoader): The data loader for fetching batches of data.
#         model (torch.nn.Module): The neural network model to train or evaluate.
#         device (torch_device): The device (CPU or GPU) to perform computations on.
#         optimizer (Optional[Optimizer]): The optimizer for training the model. If None, the runner is in evaluation mode.
#         """
#         self.device = device
#         self.model = model.to(device)
#         self.optimizer = optimizer
#         self.loader = loader
#         self.batch_size = 96
#         self.epochs = 1
#         self.score_train = []
#         self.score_test = []

#         # Assume training stage based on presence of optimizer
#         self.is_train = optimizer is not None

#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.logger.info("Initializing Runner")

#         self._prepare_data()

#     def _prepare_data(self):
#         self.logger.info("Preparing data")
#         # Your data preparation logic here
#         self.logger.info("Data preparation complete")

#     def train(self):
#         self.model.train()
#         total_loss = total_samples = 0

#         for data in self.loader:
#             data = data.to(self.device)
#             self.optimizer.zero_grad()
#             out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
#             loss = F.mse_loss(out, data.y)
#             loss.backward()
#             self.optimizer.step()
#             total_loss += float(loss) * data.num_graphs
#             total_samples += data.num_graphs

#         rmse = (total_loss / total_samples) ** 0.5
#         #self.logger.info(f"Training RMSE: {rmse:.4f}")
#         return rmse

#     @torch.no_grad()
#     def test(self, loader: GeoDataLoader):
#         self.model.eval()
#         mse = []

#         for data in loader:
#             data = data.to(self.device)
#             out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
#             l = F.mse_loss(out, data.y, reduction='none').cpu()
#             mse.append(l)

#         rmse = float(torch.cat(mse, dim=0).mean().sqrt())
#         #self.logger.info(f"Test RMSE: {rmse:.4f}")
#         return rmse

#     def fit(self, train_loader: GeoDataLoader, test_loader: GeoDataLoader, epochs: int):
#         self.epochs = epochs
#         for epoch in range(epochs):
#             train_rmse = self.train()
#             test_rmse = self.test(test_loader)
#             self.score_train.append(train_rmse)
#             self.score_test.append(test_rmse)
#             #self.logger.info(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
#         self.plot()

#     def plot(self, output_dir='plots', filename_prefix='plot'):
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         plt.figure()
#         plt.plot(range(self.epochs), self.score_train, color='goldenrod')
#         plt.plot(range(self.epochs), self.score_test, color='blue')
#         plt.xlabel('Epochs')
#         plt.ylabel('RMSE')
#         plt.legend(['train', 'test'])

#         plot_filename = os.path.join(output_dir, f'{filename_prefix}_epochs_{self.epochs}.png')
#         plt.savefig(plot_filename)
#         plt.close()
#         self.logger.info(f"Saved plot to {plot_filename}")


# ''' Some code from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py'''