import logging
import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader as GeoDataLoader  # Assuming using PyTorch Geometric

class Runner:
    def __init__(
        self,
        loader: GeoDataLoader,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: Optional[Optimizer] = None,
        epochs: int = 5
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loader = loader
        self.epochs = epochs
        self.score_train = []
        self.score_test = []
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
            total_loss = 0
            total_samples = 0
            train_loss = []

            for batch in self.loader:
                batch_graph, target = batch
                node_feats = batch_graph.ndata["hv"]
                edge_feats = batch_graph.edata["he"]

                batch_graph = batch_graph.to(self.device)
                edge_feats = edge_feats.to(self.device)
                node_feats = node_feats.to(self.device)
                target = target.to(self.device)

                predictions = self.model(batch_graph, node_feats, edge_feats)
                loss = F.mse_loss(predictions, target).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                total_loss += loss.item() * batch_graph.batch_size
                total_samples += batch_graph.batch_size

            avg_loss = total_loss / total_samples
            rmse = (avg_loss ** 0.5)
            self.logger.info(f"Epoch {epoch + 1} Training RMSE: {rmse:.4f}")
            self.score_train.append(rmse)

        return self.score_train
