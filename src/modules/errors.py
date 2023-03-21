#!/usr/bin/env python
"""
Custom errors for package
"""

from typing import List

class Error(Exception):
    """Base class for other exceptions."""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return self.message

class InvalidFile(Error):
    """Raised when file is invalid"""
    def __init__(self, file_name: int) -> None:
        self.file_name = file_name
        super().__init__(f'{self.file_name} is not a valid file')

class NotFitted(Error):
    """Raised when scaler is not fitted"""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f'{self.message} is not fitted')

class InvalidModel(Error):
    """Raised when the model type is incorrect"""
    def __init__(self, model_type: str, options = List[str]) -> None:
        self.model_type = model_type
        self.options = options
        super().__init__(f'{self.model_type} does not exists choose ({", ".join(self.options)})')

class NoProfiles(Error):
    """This TSProfiles does not contain any profiles"""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f'{self.message}')

class InvalidAttribute(Error):
    """Raised when an invalid attribute has been used"""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f'{self.message}')

class InvalidItem(Error):
    """Raised when an item is invalid"""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(f'{self.message}')

class InvalidNumberOfItem(Error):
    """Raised when an invalid number of items are used"""
    def __init__(self, items: int, actual_items: int) -> None:
        self.items = items
        self.actual_items = actual_items
        super().__init__(f'The input had {self.items} which should have been {self.actual_items}')

class InvalidType(Error):
    """Raised when the wrong type was used"""
    def __init__(self, used_type: str, actual_type: str) -> None:
        self.used_type = used_type
        self.actual_type = actual_type
        super().__init__(f'The used type was {used_type} which should have been {self.actual_type}')