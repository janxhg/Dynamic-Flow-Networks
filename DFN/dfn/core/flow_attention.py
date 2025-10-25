"""
Flow Attention Module for Dynamic Flow Networks

This module implements the Dynamic Flow Attention mechanism that replaces
traditional dot-product attention with flow-based interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def knn_search(positions: torch.Tensor, k: int) -> torch.Tensor:
    """Búsqueda de k vecinos más cercanos
    Args:
        positions: [batch_size, num_entities, pos_dim]
        k: número de vecinos
    Returns:
        neighbors: [batch_size, num_entities, k] índices de vecinos
    """
    batch_size, num_entities, pos_dim = positions.shape

    # Calcular distancias por pares
    positions_expanded = positions.unsqueeze(2)  # [batch, num_entities, 1, pos_dim]
    positions_expanded = positions_expanded.expand(-1, -1, num_entities, -1)

    distances = torch.norm(positions_expanded - positions_expanded.transpose(1, 2), dim=-1)
    # distances: [batch_size, num_entities, num_entities]

    # Excluir la entidad misma (distancia 0)
    distances = distances + torch.eye(num_entities, device=positions.device) * 1e6

    # Obtener k vecinos más cercanos
    _, neighbors = torch.topk(distances, k=k, dim=-1, largest=False)

    return neighbors


def gaussian_affinity(positions: torch.Tensor, neighbors: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Función de afinidad Gaussiana
    Args:
        positions: [batch_size, num_entities, pos_dim]
        neighbors: [batch_size, num_entities, k] índices de vecinos
        sigma: parámetro de la función Gaussiana
    Returns:
        affinities: [batch_size, num_entities, k]
    """
    batch_size, num_entities, pos_dim = positions.shape

    # Calcular distancias a vecinos
    positions_expanded = positions.unsqueeze(2)  # [batch, num_entities, 1, pos_dim]
    positions_expanded = positions_expanded.expand(-1, -1, num_entities, -1)

    distances = torch.norm(positions_expanded - positions_expanded.transpose(1, 2), dim=-1)

    # Crear máscara para solo vecinos seleccionados
    neighbor_mask = torch.zeros(batch_size, num_entities, num_entities, device=positions.device)
    for b in range(batch_size):
        for i in range(num_entities):
            neighbor_mask[b, i, neighbors[b, i]] = 1

    distances = distances * neighbor_mask

    # Aplicar función Gaussiana solo a vecinos
    affinities = torch.exp(-distances / (2 * sigma**2)) * neighbor_mask

    # Normalizar por fila
    row_sums = affinities.sum(dim=-1, keepdim=True) + 1e-8
    affinities = affinities / row_sums

    return affinities


def aggregate_flow_attention(affinities: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Agregación ponderada de valores usando afinidades
    Args:
        affinities: [batch_size, num_entities, num_entities] matriz de afinidades
        values: [batch_size, num_entities, value_dim] valores a agregar
    Returns:
        aggregated: [batch_size, num_entities, value_dim]
    """
    # affinities: [batch, num_entities, num_entities]
    # values: [batch, num_entities, value_dim]
    # aggregated: [batch, num_entities, value_dim]

    aggregated = torch.bmm(affinities, values)
    return aggregated


class DynamicFlowAttention(nn.Module):
    """Atención de Flujo Dinámico (DFA)"""

    def __init__(self, dim: int, pos_dim: int, k: int = 16, alpha: float = 0.1, sigma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.pos_dim = pos_dim
        self.k = k
        self.alpha = alpha
        self.sigma = sigma

        # Proyecciones para flujo y valores
        self.flow_projection = nn.Linear(dim, pos_dim)  # Proyectar flujo a pos_dim
        self.value_projection = nn.Linear(dim, dim)

    def forward(self, states: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aplica atención de flujo dinámico
        Args:
            states: [batch_size, num_entities, dim] estados de las entidades
            positions: [batch_size, num_entities, pos_dim] posiciones actuales
        Returns:
            context: [batch_size, num_entities, dim] contexto agregado
            new_positions: [batch_size, num_entities, pos_dim] nuevas posiciones
            flow_vectors: [batch_size, num_entities, pos_dim] vectores de flujo generados
        """
        batch_size, num_entities, dim = states.shape

        # 1. Generar vectores de flujo
        flow_vectors = self.flow_projection(states)  # [batch, num_entities, pos_dim]

        # 2. Actualizar posiciones usando flujo
        new_positions = positions + self.alpha * flow_vectors

        # 3. Buscar k vecinos más cercanos en el nuevo espacio
        neighbors = knn_search(new_positions, self.k)  # [batch, num_entities, k]

        # 4. Calcular afinidades gaussianas
        affinities = gaussian_affinity(new_positions, neighbors, self.sigma)

        # 5. Proyectar estados a valores
        values = self.value_projection(states)  # [batch, num_entities, dim]

        # 6. Agregar información de vecinos usando afinidades
        context = aggregate_flow_attention(affinities, values)  # [batch, num_entities, dim]

        return context, new_positions, flow_vectors
