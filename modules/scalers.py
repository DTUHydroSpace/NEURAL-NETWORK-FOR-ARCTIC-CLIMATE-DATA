#!/usr/bin/env python
"""
Scaling of tensors
"""
from __future__ import annotations
from typing import overload
from abc import ABC, abstractmethod
import torch
import numpy as np
from . import errors

class Scaler(ABC):
    """ Base class for scaling the data"""
    fitted: bool = False

    @staticmethod
    def to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        """ Converts input to np.ndarray"""
        if isinstance(x, np.ndarray):
            return x
        if x.is_cuda:
            x = x.cpu()
        return x.numpy()

    @abstractmethod
    def fit(self, x: np.ndarray | torch.Tensor) -> None:
        """Fits the data to the scaler"""

    @overload
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    @overload
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Scales the data"""
    
    @overload
    @abstractmethod
    def invert(self, x: np.ndarray) -> np.ndarray: ...

    @overload
    @abstractmethod
    def invert(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def invert(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Invert scales the data"""
    
    @abstractmethod
    def __repr__(self) -> str: ...

class MinMaxScaling(Scaler):
    """Min max feature scaleing torch tensor"""    
    def fit(self, x: np.ndarray | torch.Tensor) -> None:
        x = self.to_numpy(x)
        self.min_value: float = np.nanmin(x)
        self.max_value: float = np.nanmax(x)
        self.__difference: float = self.max_value - self.min_value
        self.fitted: bool = True

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.fitted:
            return (x - self.min_value)/self.__difference
        raise errors.NotFitted("MinMaxScaling")
    
    def invert(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.fitted:
            return x * self.__difference + self.min_value
        raise errors.NotFitted("MinMaxScaling")
    
    def __repr__(self) -> str:
        if self.fitted:
            return f"MinMaxScaling(min={self.min_value}, max={self.max_value})"
        raise errors.NotFitted("MinMaxScaling")

class MeanScaling(Scaler):
    """Mean feature scaleing torch tensor"""
    def fit(self, x: np.ndarray | torch.Tensor) -> None:
        x = self.to_numpy(x)
        self.mean: float = np.nanmean(x)
        self.fitted = True

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.fitted:
            if isinstance(x, torch.Tensor) and x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                x = x.float()
            return x - self.mean
        raise errors.NotFitted("MeanScaling")
    
    def invert(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.fitted:
            if isinstance(x, torch.Tensor) and x.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                x = x.float()
            return x + self.mean
        raise errors.NotFitted("MeanScaling")
    
    def __repr__(self) -> str:
        if self.fitted:
            return f"MeanScaling(mean={self.mean})"
        raise errors.NotFitted("MeanScaling")
