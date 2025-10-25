"""
Field MLP Module for Dynamic Flow Networks

This module implements the DynamicFieldMLP that updates entity states
by combining current state with contextual information from neighbors.
"""

import torch
import torch.nn as nn
from typing import Optional


class DynamicFieldMLP(nn.Module):
    """Módulo de actualización de campo dinámico"""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),  # [estado_actual, contexto]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Actualiza estados combinando información actual con contexto
        Args:
            states: [batch_size, num_entities, dim] estados actuales
            context: [batch_size, num_entities, dim] contexto de vecinos
        Returns:
            new_states: [batch_size, num_entities, dim] estados actualizados
        """
        # Concatenar estado actual con contexto de vecinos
        combined = torch.cat([states, context], dim=-1)  # [batch, num_entities, 2*dim]

        # Aplicar MLP
        new_states = self.mlp(combined)

        # Residual connection con normalización
        new_states = self.layer_norm(states + new_states)

        return new_states
