#!/usr/bin/env python
"""
Process splits of TSProfiles
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import geopandas as gp
import torch
from torch.utils.data import DataLoader
import numpy as np
from hydra import initialize, compose
from .dataHandler import TSProfiles
from . import (
    torch_util as torch_util,
    scalers as scaling
)

def split_regions(
    ts_profiles: TSProfiles,
    regions: gp.geodataframe.GeoDataFrame # type: ignore
    ) -> Dict[str, TSProfiles]:
    """ Splits the ts_profiles into different areas depending the polygons defined in regions"""
    points = gp.GeoDataFrame(ts_profiles.point, columns=['geometry']).set_crs(4326)
    points_within = gp.sjoin(
        points,
        gp.GeoDataFrame(regions.geometry),
        op='within'
    )
    ts_areas = {}
    for idx in points_within.index_right.unique():
        ts_areas[regions.Name[idx]] = ts_profiles[points_within.index[points_within.index_right == idx].to_list()]
    ts_areas['all'] = ts_profiles
    return ts_areas

def split_data_set(
    ts_areas: Dict[str, TSProfiles],
    training_end: int,
    validation_end: int,
    min_time: int = 0,
    max_time: int = 5000
    ) -> Tuple[Dict[str, TSProfiles], Dict[str, TSProfiles], Dict[str, TSProfiles], List[str]]:
    """
    Splits the dataset into training, testing, validation for each region.
    The split is made on the training_end and validation_end
    (These two values are the years the splits happend)
    """
    train, val, test = {}, {}, {}
    for area, profiles in ts_areas.items():
        profile_times = profiles.time
        train[area] = profiles[(profile_times <= training_end) & (profile_times > min_time)]
        val[area] = profiles[(training_end < profile_times) & (profile_times <= validation_end)]
        test[area] = profiles[(profile_times <= max_time) & (profile_times > validation_end)]
    return train, val, test, list(ts_areas)

def print_size(train: Dict[str, TSProfiles], val: Dict[str, TSProfiles], test: Dict[str, TSProfiles], key: str) -> None:
    """ Prints the size of the chosen dataset"""
    print(f'Data set over {key}')
    print(f'{"Training:":<15}{len(train[key]):>20,d}')
    print(f'{"Validation:":<15}{len(val[key]):>20,d}')
    print(f'{"Testing:":<15}{len(test[key]):>20,d}')

def process_data(
    train: Dict[str, TSProfiles],
    val: Dict[str, TSProfiles],
    test: Dict[str, TSProfiles],
    bathymetri_lat: np.ndarray,
    bathymetri_lon: np.ndarray,
    bathymetri_topography: np.ndarray,
    scalers: List[scaling.Scaler]
    ) -> Tuple[List[scaling.Scaler], DataLoader, DataLoader, DataLoader]:
    """ Processes a specific region by scaling all the datasets and converting the dicts to dataloaders"""
    with initialize(config_path="../config"):
        cfg = compose(config_name="experiment").experiment

    bathymetri_data = [bathymetri_lat, bathymetri_lon, bathymetri_topography]
    max_length = train[cfg.area].profile_length.max()

    scalers, train_features, train_labels = torch_util.to_torch(train[cfg.area], max_length, scalers, *bathymetri_data)
    _, val_features, val_labels = torch_util.to_torch(val[cfg.area], max_length, scalers, *bathymetri_data)
    _, test_features, test_labels = torch_util.to_torch(test[cfg.area], max_length, scalers, *bathymetri_data)

    train_loader = torch_util.to_dataloader(torch.cat(train_features,2), torch.cat(train_labels,2), batch_size=cfg.batch_size)
    val_loader = torch_util.to_dataloader(torch.cat(val_features,2), torch.cat(val_labels,2), batch_size=cfg.batch_size)
    test_loader = torch_util.to_dataloader(torch.cat(test_features,2), torch.cat(test_labels,2), batch_size=cfg.batch_size)
    return scalers, train_loader, val_loader, test_loader