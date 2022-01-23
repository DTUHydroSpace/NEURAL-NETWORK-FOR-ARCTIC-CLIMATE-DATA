#!/usr/bin/env python
"""
Transformation of data to correct formats
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Any, Type
from . import scalers
from .dataHandler import TSProfiles

class Dict(dict):
    """ Helper class for avoiding using strings in the other classes"""
    def __getattr__(self, name: str) -> Any: return self[name]
    def __setattr__(self, name: str, value: Any) -> None: self[name] = value
    def __delattr__(self, name: str) -> None: del self[name]

def get_temperatur(
    profile_times: np.ndarray[np.float64, np.dtype[np.float64]],
    sst_times: np.ndarray[np.float64, np.dtype[np.float64]],
    sst: np.ndarray[np.float64, np.dtype[np.float64]]
    ) -> torch.Tensor:
    """ Get temperatur profile for the correct time."""
    closest_times = np.argmin(abs(profile_times - sst_times[:, np.newaxis]),axis=0)
    return torch.Tensor(sst[closest_times,:,:])

def get_bathymetri(
    lat: np.ndarray[np.float64, np.dtype[np.float64]],
    lon: np.ndarray[np.float64, np.dtype[np.float64]],
    bathymetri_lat: np.ndarray[np.float64, np.dtype[np.float64]],
    bathymetri_lon: np.ndarray[np.float64, np.dtype[np.float64]],
    bathymetri_topography: np.ndarray[np.float64, np.dtype[np.float64]]
    ) -> np.ndarray[np.float64, np.dtype[np.float64]]:
    """ Get bathymetri profile for the correct time."""
    min_lat_idx = np.argmin(abs(lat - bathymetri_lat[:, np.newaxis]), axis=0)
    min_lon_idx = np.argmin(abs(lon - bathymetri_lon[:, np.newaxis]), axis=0)
    return bathymetri_topography[min_lat_idx, min_lon_idx]

def feature_scale(data: np.ndarray, scaler: scalers.Scaler | Type[scalers.Scaler], max_length: int = None) -> Tuple[scalers.Scaler, torch.Tensor]:
    """ Feature scale input and converts it to a tensor"""
    if data.dtype == 'object':
        data = pad_tensors(data, max_length=max_length).numpy()
    if not isinstance(scaler, scalers.Scaler):
        fitted_scaler: scalers.Scaler = scaler()
        fitted_scaler.fit(data)
    elif not scaler.fitted:
        scaler.fit(data)
        fitted_scaler: scalers.Scaler = scaler
    else:
        fitted_scaler: scalers.Scaler = scaler
    data_tensor = torch.tensor(data)
    data_fitted = fitted_scaler(data_tensor)
    return fitted_scaler, data_fitted

def pad_tensors(numpy_list: np.ndarray, padding_value: Any = np.nan, max_length: int = None) -> torch.Tensor:
    """ Convert numpy arrays to tensors with padding"""
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in numpy_list],
        batch_first=True,
        padding_value=padding_value
    )
    if max_length is None:
        return padded
    return torch.nn.functional.pad( # type: ignore
        padded,
        pad=(0,max_length - padded.size(1)),
        mode='constant',
        value=padding_value
    )

def fit_scalers(
    dataset: TSProfiles,
    max_length:int,
    scalers: List[Type[scalers.Scaler]],
    bathymetri_lat: np.ndarray,
    bathymetri_lon: np.ndarray,
    bathymetri_topography: np.ndarray
) -> List[scalers.Scaler]:
    """ Fits scalers to data"""
    salinity: List[np.ndarray] = dataset.salinity
    temperatur: List[np.ndarray] = dataset.temperatur

    features_scaling = [
        dataset.lat[:, np.newaxis],# Lat
        dataset.lon[:, np.newaxis],# Lon
        dataset.time[:, np.newaxis] // 1, # Year
        dataset.time[:, np.newaxis] % 1, # Decimal year
        np.array(list(map(lambda x: x[0], salinity)))[:, np.newaxis], # Surface salinity
        np.array(list(map(lambda x: x[0], temperatur)))[:, np.newaxis], # Surface temperatur
        np.array(list(map(lambda x: x[0], dataset.depth)))[:, np.newaxis], # "Surface" depth
        get_bathymetri(dataset.lat, dataset.lon, bathymetri_lat, bathymetri_lon, bathymetri_topography)[:, np.newaxis], # Bathymetri
        salinity, # Salinity profile
        dataset.temperatur # Temperature profile
    ]
    

    save_scalers: List[scalers.Scaler] = []
    for i, f in enumerate(features_scaling):
        scaler = scalers[i]()
        if f.dtype == 'object':
            f: np.ndarray = pad_tensors(f, max_length=max_length).numpy()
        scaler.fit(f)
        save_scalers.append(scaler)
    return save_scalers

def to_torch(
    dataset: TSProfiles,
    max_length:int,
    scalers: List[scalers.Scaler],
    bathymetri_lat: np.ndarray,
    bathymetri_lon: np.ndarray,
    bathymetri_topography: np.ndarray
) -> Tuple[List[scalers.Scaler], List[torch.Tensor], List[torch.Tensor]]:
    """ Converts TSProfiles to scaled torch tensors (scalers, x, y)"""
    salinity: List[np.ndarray] = dataset.salinity
    temperatur: List[np.ndarray] = dataset.temperatur

    features_scaling = [
        dataset.lat[:, np.newaxis],# Lat
        dataset.lon[:, np.newaxis],# Lon
        dataset.time[:, np.newaxis] // 1, # Year
        dataset.time[:, np.newaxis] % 1, # Decimal year
        np.array(list(map(lambda x: x[0], salinity)))[:, np.newaxis], # Surface salinity
        np.array(list(map(lambda x: x[0], temperatur)))[:, np.newaxis], # Surface temperatur
        np.array(list(map(lambda x: x[0], dataset.depth)))[:, np.newaxis], # "Surface" depth
        get_bathymetri(dataset.lat, dataset.lon, bathymetri_lat, bathymetri_lon, bathymetri_topography)[:, np.newaxis] # Bathymetri
    ]
    
    labels_scaling = [
        salinity, # Salinity profile
        dataset.temperatur # Temperature profile
    ]

    save_features: List[torch.Tensor] = []
    save_scalers: List[scalers.Scaler] = []
    for i, f in enumerate(features_scaling):
        scaler, feature = feature_scale(f, max_length=max_length, scaler=scalers[i])
        save_scalers.append(scaler)
        save_features.append(feature.unsqueeze(0))
    
    save_labels: List[torch.Tensor] = []
    for i, f in enumerate(labels_scaling):
        j = i+len(features_scaling)
        scaler, label = feature_scale(f, max_length=max_length, scaler=scalers[j])
        save_scalers.append(scaler)
        save_labels.append(label.squeeze(0).unsqueeze(-1))

    return save_scalers, save_features, save_labels


def to_dataloader(features: torch.Tensor, labels: torch.Tensor, batch_size: int = 16) -> DataLoader:
    """ Converts dataset to dataloaders"""
    return DataLoader(
        TensorDataset(features.squeeze(0).cuda(), labels.squeeze(0).cuda()),
        batch_size=batch_size,
        shuffle=False
    )