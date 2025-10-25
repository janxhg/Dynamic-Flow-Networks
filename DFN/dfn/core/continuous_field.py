"""
Continuous Field Module for Dynamic Flow Networks

This module implements the ContinuousField class that maps discrete inputs
to continuous fields and extracts adaptive entities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ContinuousField(nn.Module):
    """Mapea entradas discretas a un campo continuo F(x)"""

    def __init__(self, input_dim: int, field_dim: int, num_entities: int, pos_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, field_dim)
        self.field_mapper = nn.Sequential(
            nn.Linear(field_dim, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, pos_dim + input_dim + 1)  # p, s, w
        )
        self.state_projection = nn.Linear(field_dim, input_dim)  # Proyectar estados a dim
        self.num_entities = num_entities
        self.pos_dim = pos_dim
        self.field_dim = field_dim
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extrae entidades continuas del campo
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            positions: [batch_size, num_entities, pos_dim]
            states: [batch_size, num_entities, d]
            weights: [batch_size, num_entities, 1]
        """
        batch_size, seq_len, input_dim = x.shape

        # Proyectar a espacio de campo
        field = self.input_proj(x)  # [batch, seq_len, field_dim]

        # Crear campo continuo expandido
        field_expanded = field.unsqueeze(1).expand(-1, seq_len, -1, -1)
        field_expanded = field_expanded.reshape(batch_size * seq_len, seq_len, -1)

        # Mapear a entidades (posiciones, estados, pesos)
        entity_features = self.field_mapper(field)  # [batch, seq_len, field_dim*3]

        # Muestreo adaptativo basado en densidad de información
        importance = torch.norm(entity_features, dim=-1)  # [batch, seq_len]
        importance = F.softmax(importance, dim=-1)

        # Seleccionar top-k entidades más importantes
        _, top_indices = torch.topk(importance, min(self.num_entities, seq_len), dim=-1)

        positions = []
        states = []
        weights = []

        for b in range(batch_size):
            batch_indices = top_indices[b]

            # Extraer características de las entidades seleccionadas
            selected_features = entity_features[b][batch_indices]  # [num_entities, pos_dim + field_dim + 1]

            # Separar en posiciones, estados y pesos
            total_dim = selected_features.size(-1)
            pos = selected_features[:, :self.pos_dim]
            state = selected_features[:, self.pos_dim:self.pos_dim + self.field_dim]
            weight = selected_features[:, -1:]

            # Proyectar estados a la dimensión del modelo
            state = self.state_projection(state)  # [num_entities, input_dim]

            positions.append(pos.unsqueeze(0))
            states.append(state.unsqueeze(0))
            weights.append(weight.unsqueeze(0))

        positions = torch.cat(positions, dim=0)  # [batch, num_entities, pos_dim]
        states = torch.cat(states, dim=0)      # [batch, num_entities, d]
        weights = torch.cat(weights, dim=0)    # [batch, num_entities, 1]

        return positions, states, weights
