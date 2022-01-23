#!/usr/bin/env python
"""
Temperature / Salinity models
"""
from hydra import initialize, compose
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
import json
from dataclasses import dataclass

from . import (
    errors as errors,
    torch_util as torch_util
)

@dataclass
class TrainingLoss:
    """Class for training loss"""
    loss: float
    epoch: int
    learning_rate: float

@dataclass
class ValidationLoss:
    """Class for validation loss"""
    loss: float
    epoch: int
    medae: float # Median absolute error

class TSNet(torch.nn.Module):
    def __init__(self, rnn_setting: torch_util.Dict, linear_settings: List[torch_util.Dict], model_type: str, dropout:float = 0.0) -> None:
        super().__init__()
        # Check input
        if model_type.lower() == 'salinity':
            self.idx = 0
        elif model_type.lower() == 'temperature':
            self.idx = 1
        else:
            raise errors.InvalidModel(model_type, ["Salinity", "Temperature"])

        # Save settings
        self.rnn_setting = rnn_setting
        self.linear_settings = linear_settings
        self.dropout = dropout
        self.model_type = model_type
        
        # RNN
        self.drop = torch.nn.Dropout(p=self.dropout)
        self.rnn = torch.nn.GRU(
            self.rnn_setting.input_size,
            self.rnn_setting.output_size,
            self.rnn_setting.n_layers,
            dropout=self.dropout
        ).double()

        # fully connected
        self.fc = torch.nn.Sequential(
            *[torch.nn.Linear(ls.input_size, ls.output_size).double() for ls in self.linear_settings]
        )

    def is_cuda(self) -> bool:
        """ Check if the model is cuda or not"""
        return next(self.parameters()).is_cuda
        
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """ Forward pass"""
        yn = self.drop(input_features)
        hn = None
        output_rnn, _ = self.rnn(yn, hn)
        return self.fc(self.drop(output_rnn))

    def train_epoch(self, epoch: int, dataloader: DataLoader, optimizer, scheduler, criterion) -> Tuple[float, float]:
        """ Trains the model for one epoch"""
        self.train()
        iters = len(dataloader)
        train_losses = []
        for idx, (features, labels) in enumerate(dataloader):
            # Setup data
            if not self.is_cuda():
                features, labels = features.cpu(), labels.cpu()
            y_true = labels[:,:,self.idx]

            # model output
            optimizer.zero_grad()
            output = self(features.unsqueeze(1)).squeeze(1)

            # Padding and true value
            remove_padding = ~torch.isnan(y_true)
            y_true_non_padded = y_true[remove_padding]
            if not self.is_cuda():
                y_true_non_padded = y_true_non_padded.cpu()

            # Loss and optimize
            loss = criterion(output[remove_padding], y_true_non_padded)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + idx / iters)
            
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        lr = optimizer.param_groups[0]["lr"]
        return train_loss, lr
    
    def predict(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Evals the model for one epoch"""
        self.eval()
        y_pred_list = []
        y_true_list = []
        for features, labels in dataloader:
            # Setup data
            if not self.is_cuda():
                features, labels = features.cpu(), labels.cpu()
            y_true = labels[:,:,self.idx]

            # model output
            output = self(features.unsqueeze(1)).squeeze(1)

            # Padding and true value
            remove_padding = ~np.isnan(y_true.cpu().numpy())
            y_true_non_padded = y_true[remove_padding]
            if not self.is_cuda():
                y_true_non_padded = y_true_non_padded.cpu()
            y_pred_list.append(output[remove_padding])
            y_true_list.append(y_true_non_padded)      
        return torch.cat(y_pred_list, 0), torch.cat(y_true_list, 0)

def print_epoch(train_loss: TrainingLoss, val_loss: ValidationLoss, total_epoch: int, model_type: str) -> None:
    """ Prints the statistics from a losses dict"""
    total_epoch_len = len(str(total_epoch)) - len(str(val_loss.epoch+1))
    header = f"[{' '*total_epoch_len}{val_loss.epoch+1}/{total_epoch:<3}] {model_type}"
    train_print =  f"""Loss {train_loss.loss:.5f}, lr {train_loss.learning_rate:.3f}"""
    val_print =  f"""Loss {val_loss.loss:.5f}, MedAE {val_loss.medae:.3f}"""
    print(f"{header} Training {train_print} | Validation {val_print}")

def load_model(train_loader: DataLoader) -> TSNet:
    """ Create a model based on the dataloader and loads in a trained model"""
    with initialize(config_path="../config"):
        cfg = compose(config_name="experiment").experiment
    features, labels = next(iter(train_loader))

    # Setup setting for the model
    rnn_setting = torch_util.Dict(
        input_size=features.size(1),
        output_size=labels.size(1),
        n_layers=cfg.gru_layers
    )
    if cfg.linear_layers and isinstance(cfg.linear_layers, list):
        dims = cfg.linear_layers
        dims.insert(0, labels.size(1))
        dims.append(labels.size(1))
    else:
        dims = [labels.size(1), labels.size(1)]
    linear_settings=[torch_util.Dict(input_size = d1, output_size=d2) for d1, d2 in zip(dims[:-1], dims[1:])]

    # Init model
    model = TSNet(
        rnn_setting=rnn_setting,
        linear_settings=linear_settings,
        dropout=cfg.dropout,
        model_type=cfg.model_type
    ).cuda()

    # load model
    model.load_state_dict(torch.load(f"models//{cfg.model_name}.pt"))
    model.eval()
    return model

def setup_model(train_loader: DataLoader) -> TSNet:
    """ Create a model based on the dataloader and the settings in the config file."""
    with initialize(config_path="../config"):
        cfg = compose(config_name="experiment").experiment
    features, labels = next(iter(train_loader))

    # Setup setting for the model
    rnn_setting = torch_util.Dict(
        input_size=features.size(1),
        output_size=labels.size(1),
        n_layers=cfg.gru_layers
    )
    if cfg.linear_layers and isinstance(cfg.linear_layers, list):
        dims = cfg.linear_layers
        dims.insert(0, labels.size(1))
        dims.append(labels.size(1))
    else:
        dims = [labels.size(1), labels.size(1)]
    linear_settings=[torch_util.Dict(input_size = d1, output_size=d2) for d1, d2 in zip(dims[:-1], dims[1:])]

    # Init model
    return TSNet(
        rnn_setting=rnn_setting,
        linear_settings=linear_settings,
        dropout=cfg.dropout,
        model_type=cfg.model_type
    ).cuda()

def train_model(train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[TrainingLoss], List[ValidationLoss]]:
    """ Train chosen model based on setting in the config file"""
    with initialize(config_path="../config"):
        cfg = compose(config_name="experiment").experiment
    model = setup_model(train_loader=train_loader)

    # Loss and optimizers
    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.T_0,
        T_mult=cfg.T_mult,
        eta_min=0.001,
        last_epoch=-1
    )

    # Save values
    training_loss, validation_loss = [], []

    # Print before start of training
    print(f"Model:\n\tRNN:\n\t\tInput {model.rnn_setting.input_size}, Output {model.rnn_setting.output_size}, Layers {model.rnn_setting.n_layers}")
    print(f"\tLinear:\n\t\tInput {model.linear_settings[0].input_size}, Output {model.linear_settings[-1].output_size}, Layers {len(model.linear_settings)}")
    print(f"Optimizer:\n\tlr {cfg.learning_rate}, momentum {cfg.momentum} weight decay {cfg.weight_decay}\n")
    for i in range(cfg.epochs):
        # Training
        train_loss, lr = model.train_epoch(i, train_loader, optimizer, scheduler, criterion)
        t_loss = TrainingLoss(loss=train_loss, epoch=i, learning_rate=lr)
        training_loss.append(t_loss)
        
        if (i+1) % cfg.print_every == 0 or i==0 or i==cfg.epochs:
            # Validation
            y_pred, y_true = model.predict(val_loader)
            val_loss = criterion(input=y_pred, target=y_true).item()
            medae = torch.median(abs(y_pred - y_true)).item()
            v_loss = ValidationLoss(loss=val_loss, epoch=i, medae=medae)
            validation_loss.append(v_loss)

            # Print
            print_epoch(t_loss, v_loss, cfg.epochs, cfg.model_type)

    # Save model
    torch.save(model.state_dict(), f"models//{cfg.model_name}.pt")
    with open(f"models//{cfg.model_name}.json", 'w') as fp:
        json.dump(
            {
                'train': [t.__dict__ for t in training_loss],
                'validation': [v.__dict__ for v in validation_loss]
            },
            fp
        )
    return training_loss, validation_loss