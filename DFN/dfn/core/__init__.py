"""
Core components for Dynamic Flow Networks

This module contains the fundamental building blocks of the DFN architecture:
- ContinuousField: Maps discrete inputs to continuous fields
- DynamicFlowAttention: Flow-based attention mechanism
- FlowNorm: Normalization for flow stabilization
- DynamicFieldMLP: Entity state updates
- PersistentFieldMemory: Long-term context memory
- DFNLayer: Main DFN layer combining all components
"""

from .continuous_field import ContinuousField
from .flow_attention import DynamicFlowAttention, knn_search, gaussian_affinity, aggregate_flow_attention
from .normalization import FlowNorm
from .field_mlp import DynamicFieldMLP
from .memory import PersistentFieldMemory
from .dfn_layer import DFNLayer

__all__ = [
    "ContinuousField",
    "DynamicFlowAttention",
    "FlowNorm",
    "DynamicFieldMLP",
    "PersistentFieldMemory",
    "DFNLayer",
    # Utility functions
    "knn_search",
    "gaussian_affinity",
    "aggregate_flow_attention"
]
