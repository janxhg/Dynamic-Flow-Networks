"""
Models Module for Dynamic Flow Networks

This module contains complete DFN model implementations for different tasks:
- DFNPureModel: Pure DFN without vocabulary or sequence length constraints
"""

from .dfn_pure import DFNPureModel

__all__ = [
    "DFNPureModel"
]
