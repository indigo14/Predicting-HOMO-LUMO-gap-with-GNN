import logging
import optuna
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import AttentiveFP
from src.runner import Runner
from sklearn.metrics import r2_score
from typing import Dict

logger = logging.getLogger(__name__)

def objective(trial, train_loader, test_loader):
    # Define the hyperparameters to tune
    
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    

    # Log the hyperparameters
    logger.info(f"Trial {trial.number}:  lr={lr}, weight_decay={weight_decay}")
    
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

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPNNPredictor(**model_params).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.MSELoss(reduction='none')


    
    

    train_runner = Runner(train_loader, val_loader, model, device, optimizer)
    test_runner = Runner(test_loader, model, device)
    
    # Train the model
    train_runner.fit(train_loader, test_loader, epochs=50)
    
    # Evaluate the model
    test_rmse = test_runner.test(test_loader)
    
    # Log the result
    logger.info(f"Trial {trial.number} Test RMSE: {test_rmse:.4f}")
    
    return test_rmse

def tune_hyperparameters(train_loader: DataLoader, test_loader: DataLoader) -> Dict:
    def wrapped_objective(trial):
        return objective(trial, train_loader, test_loader)
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(wrapped_objective, n_trials=400)
    
    # Log the best hyperparameters
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    # Visualize the optimization history
    optuna.visualization.plot_optimization_history(study).show()
    
    # Visualize the hyperparameter importances
    optuna.visualization.plot_param_importances(study).show()
    
    return study.best_params

if __name__ == "__main__":
    # Assuming train_loader and test_loader are defined globally
    best_params = tune_hyperparameters(train_loader, test_loader)
    logger.info(f"Best parameters: {best_params}")
