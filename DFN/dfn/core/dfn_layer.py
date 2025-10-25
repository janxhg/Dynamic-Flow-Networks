"""
DFN Layer Module for Dynamic Flow Networks

This module implements the main DFNLayer that combines all core components
into a complete processing layer for the DFN architecture.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .continuous_field import ContinuousField
from .flow_attention import DynamicFlowAttention
from .normalization import FlowNorm
from .field_mlp import DynamicFieldMLP
from .memory import PersistentFieldMemory


class DFNLayer(nn.Module):
    """Capa principal de Dynamic Flow Networks"""

    def __init__(self, dim: int, pos_dim: int, k: int = 16, alpha: float = 0.1, sigma: float = 1.0,
                 memory_size: int = 64, beta: float = 0.9, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.k = k
        self.alpha = alpha
        self.sigma = sigma

        # Componentes principales
        self.flow_attention = DynamicFlowAttention(dim, pos_dim, k, alpha, sigma)
        self.field_mlp = DynamicFieldMLP(dim, dropout=dropout)
        self.memory = PersistentFieldMemory(dim, memory_size, beta)
        self.flow_norm = FlowNorm()

    def forward(self, states: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paso forward de la capa DFN
        Args:
            states: [batch_size, num_entities, dim] estados actuales
            positions: [batch_size, num_entities, pos_dim] posiciones actuales
        Returns:
            new_states: [batch_size, num_entities, dim] estados actualizados
            new_positions: [batch_size, num_entities, pos_dim] nuevas posiciones
        """
        # 1. Atención de flujo dinámico
        context, new_positions, flow_vectors = self.flow_attention(states, positions)

        # 2. Incorporar contexto de memoria persistente
        memory_context = self.memory(states, positions)
        context = context + memory_context

        # 3. Actualización de estados usando MLP
        new_states = self.field_mlp(states, context)

        # 4. Normalización de flujo
        new_states = self.flow_norm(new_states, flow_vectors)

        return new_states, new_positions

    def reset_memory(self):
        """Reinicia la memoria persistente"""
        self.memory.reset_memory()
