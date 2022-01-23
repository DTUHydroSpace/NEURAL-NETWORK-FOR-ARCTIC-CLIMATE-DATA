#!/usr/bin/env python
"""
Different plots of the data
"""
from __future__ import annotations
from torch.utils.data import DataLoader
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.feature as cfeature
import numpy as np
import torch
from . import torch_util as torch_util

def get_earth_featurs(resol: str = '50m') -> Tuple[
    cfeature.NaturalEarthFeature,
    cfeature.NaturalEarthFeature,
    cfeature.NaturalEarthFeature,
    cfeature.NaturalEarthFeature,
    cfeature.NaturalEarthFeature
]:
    """ Returns earths features with a given resolution"""
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    lands = cfeature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    lakes = cfeature.NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
    rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='b', facecolor='none')
    return bodr, lands, ocean, lakes, rivers

def plot_salinity_model_depth(
    y_pred: torch.Tensor, chosen_depth: int,
    depths: np.ndarray, loader: DataLoader, scaler: torch_util.Dict,
    xlim: List[float] | List[int], ylim: List[float] | List[int], clim: List[float] | List[int], log_scale: bool = True
    ) -> None:
    """ Plots the salinity difference at a given depth"""
    # Get valid data
    bool_depth = np.concatenate(depths) == chosen_depth
    diff_salinity = scaler.salinity.invert(y_pred[bool_depth]).cpu().detach().numpy()
    # No values
    no_values = [(d == chosen_depth).any() for d in depths]
    
    lat = scaler.lat.invert(loader.dataset[:][0][:,0]).cpu().numpy()[no_values]
    lon = scaler.lon.invert(loader.dataset[:][0][:,1]).cpu().numpy()[no_values]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14,10))
    ax_main, ax_color, ax_hist, cbar = plot_color_map(lat, lon, diff_salinity, xlim, ylim, clim, log_scale, ax)
    ax_main.set_title(f"Model salinity at {chosen_depth}m")
    ax_main.set_xlabel(r"Longitude [$\degree$]")
    ax_main.set_ylabel(r"Latitude [$\degree$]")
    cbar.ax.set_ylabel('Salinity [psu]')
    ax_hist.set_ylabel('Salinity [psu]')
    ax_hist.set_xlabel('Count')
    ax_hist.set_title("Count of the\n salinity")
    plt.show()

def plot_salinity_depth(
    y_true: torch.Tensor, chosen_depth: int,
    depths: np.ndarray, loader: DataLoader, scaler: torch_util.Dict,
    xlim: List[float] | List[int], ylim: List[float] | List[int], clim: List[float] | List[int], log_scale: bool = True
    ) -> None:
    """ Plots the salinity difference at a given depth"""
    # Get valid data
    bool_depth = np.concatenate(depths) == chosen_depth
    diff_salinity = scaler.salinity.invert(y_true[bool_depth]).cpu().detach().numpy()
    # No values
    no_values = [(d == chosen_depth).any() for d in depths]
    
    lat = scaler.lat.invert(loader.dataset[:][0][:,0]).cpu().numpy()[no_values]
    lon = scaler.lon.invert(loader.dataset[:][0][:,1]).cpu().numpy()[no_values]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14,10))
    ax_main, ax_color, ax_hist, cbar = plot_color_map(lat, lon, diff_salinity, xlim, ylim, clim, log_scale, ax)
    ax_main.set_title(f"Salinity at {chosen_depth}m")
    ax_main.set_xlabel(r"Longitude [$\degree$]")
    ax_main.set_ylabel(r"Latitude [$\degree$]")
    cbar.ax.set_ylabel('Salinity [psu]')
    ax_hist.set_ylabel('Salinity [psu]')
    ax_hist.set_xlabel('Count')
    ax_hist.set_title("Count of the\n salinity")
    plt.show()


def plot_different_salinity_depth(
    y_pred: torch.Tensor, y_true: torch.Tensor, chosen_depth: int,
    depths: np.ndarray, loader: DataLoader, scaler: torch_util.Dict,
    xlim: List[float] | List[int], ylim: List[float] | List[int], clim: List[float] | List[int], log_scale: bool = True
    ) -> None:
    """ Plots the salinity difference at a given depth"""
    # Get valid data
    bool_depth = np.concatenate(depths) == chosen_depth
    diff_salinity = abs(y_pred[bool_depth] - y_true[bool_depth]).cpu().detach().numpy()
    # No values
    no_values = [(d == chosen_depth).any() for d in depths]
    
    lat = scaler.lat.invert(loader.dataset[:][0][:,0]).cpu().numpy()[no_values]
    lon = scaler.lon.invert(loader.dataset[:][0][:,1]).cpu().numpy()[no_values]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14,10))
    ax_main, ax_color, ax_hist, cbar = plot_color_map(lat, lon, diff_salinity, xlim, ylim, clim, log_scale, ax)
    ax_main.set_title(f"Absolute salinity difference at {chosen_depth}m\n(between model and known values)")
    ax_main.set_xlabel(r"Longitude [$\degree$]")
    ax_main.set_ylabel(r"Latitude [$\degree$]")
    cbar.ax.set_ylabel('Absolute salinity difference [psu]')
    ax_hist.set_ylabel('Absolute salinity difference [psu]')
    ax_hist.set_xlabel('Count')
    ax_hist.set_title("Count of the\nabsolute salinity difference")
    
    plt.show()

def plot_color_map(
    lat: np.ndarray, lon: np.ndarray, z: np.ndarray,
    xlim: List[float] | List[int], ylim: List[float] | List[int], clim: List[float] | List[int],
    log_scale: bool = True, ax: plt.Axes = None
    ) -> Tuple[plt.Axes, plt.Axes, plt.Axes, plt.colorbar]: # type: ignore
    """ Plots scatter plot on a map with a colorbar and histogram"""
    # Plots
    _, lands, _, _, _ = get_earth_featurs()
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,10))

    for land in lands.geometries():
        for geom in land.geoms: # type: ignore
            ax.plot(*geom.exterior.xy, alpha=1, c='k')
    # Normalize colorbar
    if log_scale:
        scale = colors.LogNorm(clim[0], clim[1])
    else:
        scale = colors.Normalize(clim[0], clim[1])
    
    divider = make_axes_locatable(plt.gca())
    axBar = divider.append_axes("right", size='5%', pad='5%')
    axHist = divider.append_axes("right", size='50%', pad='15%')
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cm = plt.cm.get_cmap('viridis')
    sc = ax.scatter(
        lon, lat, c=z, # type: ignore
        s=10, cmap=cm, norm=scale
    )
    cbar = plt.colorbar(sc, cax=axBar)
    
    norm = colors.Normalize(clim[0], clim[1])

    # Get hist data
    if log_scale:
        _, bins, _ = plt.hist(z, bins=50)
        bin_size = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins)) # type: ignore
    else:
        bin_size = np.linspace(clim[0], clim[1], 50)
    N, bins, patches = axHist.hist(z, bins=bin_size, orientation='horizontal')
    axHist.grid()
    axHist.set_ylim(clim)
    if log_scale:
        #axHist.set_xscale('log')
        axHist.set_yscale('log')
    axHist.yaxis.tick_right()
    axHist.yaxis.set_label_position("right")
    # set a color for every bar (patch) according 
    # to bin value from normalized min-max interval
    for single_bin, patch in zip(bins, patches):
        color = cm(scale(single_bin))
        patch.set_facecolor(color)
    
    return ax, axBar, axHist, cbar


def show_plots(y_pred: torch.Tensor, y_true: torch.Tensor, depths: np.ndarray, loader: DataLoader, scaler: torch_util.Dict, figures: int = 10):
    """ Plots predicted values"""
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    length = 0
    for i in range(figures):
        depth = depths[i]
        cur_length = len(depth)
        predicted_values = y_pred[length:cur_length+length]
        true_values = y_true[length:cur_length+length]
        length += len(depth)

        lat = scaler.lat.invert(loader.dataset[i][0][0])
        lon = scaler.lon.invert(loader.dataset[i][0][1])
        time = scaler.year.invert(loader.dataset[i][0][2]) + scaler.decimal_year.invert(loader.dataset[i][0][3])

        plt.figure()
        plt.title(f"(lon, lat) = ({lon.item()}, {lat.item()}),\ntime = {time.item()}")
        plt.plot(scaler.salinity.invert(predicted_values[::-1]), depth[::-1], label="Predicted", marker='.')
        plt.plot(scaler.salinity.invert(true_values[::-1]), depth[::-1], label="True", marker='.')
        plt.legend()
       # plt.xlim([25,35])
        plt.ylim([105,-5])
        plt.grid()
        plt.xlabel("Salinity [psu]")
        plt.ylabel("Depth [m]")
        plt.show()

def show_all(y_pred: torch.Tensor, y_true: torch.Tensor, depths: np.ndarray, loader: DataLoader, scaler: torch_util.Dict, profiles: int = 10, ax: plt.Axes = None) -> plt.Axes:
    """ Plots all profiles together for both true values and predicted values"""
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    length = 0
    
    t_v, p_v, d_list = [], [], []
    for i in range(profiles):
        depth = depths[i]
        cur_length = len(depth)
        p_v.append(y_pred[length:cur_length+length])
        t_v.append(y_true[length:cur_length+length])
        length += len(depth)
        d_list.append(depth)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,10))
    for i, (t, d) in enumerate(zip(t_v, d_list)):
        ax.plot(scaler.salinity.invert(t), d, label="Observed" if i==0 else '', color='blue', alpha=0.1)
    for i, (p, d) in enumerate(zip(p_v, d_list)):
        ax.plot(scaler.salinity.invert(p), d, label="Predicted" if i==0 else '', color='red', alpha=0.1)
    ax.legend()
    ax.set_xlim([20,35])
    ax.set_ylim([105,-5])
    ax.set_ylabel("Depth [m]")
    ax.set_xlabel("Salinity [psu]")
    ax.grid()
    return ax
    
def plot_error_depth(y_pred: torch.Tensor, y_true: torch.Tensor, depths: np.ndarray, ax: plt.Axes = None, legend: str=None, color: str=None) -> plt.Axes:
    """ Plots the error as a function of depth"""
    unique_depths = np.unique(np.concatenate(depths))
    all_depths = np.concatenate(depths)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,10))
    
    save_diff = []
    for d in unique_depths:
        bool_depth = all_depths == d
        difference = (y_pred[bool_depth] - y_true[bool_depth]).cpu().detach().numpy()
        rmsd = np.sqrt(sum(difference**2)/len(difference))
        
        save_diff.append(rmsd)
    ax.plot(save_diff, unique_depths, label=legend, color=color)
    return ax