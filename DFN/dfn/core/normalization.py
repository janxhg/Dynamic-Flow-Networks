"""
Normalization Module for Dynamic Flow Networks

This module implements FlowNorm, a normalization technique specifically
designed to stabilize the dynamics of flow-based attention mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional


class FlowNorm(nn.Module):
    """Normalización de flujo para estabilizar magnitudes"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, flow_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normaliza usando la magnitud esperada de los vectores de flujo
        Args:
            x: tensor a normalizar [batch, ..., dim]
            flow_vectors: vectores de flujo para calcular norma [batch, ..., dim]
        Returns:
            normalized_x: tensor normalizado
        """
        if flow_vectors is None:
            # Usar el tensor mismo como flujo si no se proporciona
            flow_vectors = x

        # Calcular norma esperada del flujo
        flow_magnitude = torch.norm(flow_vectors, dim=-1, keepdim=True)
        expected_magnitude = torch.mean(flow_magnitude ** 2, dim=-2, keepdim=True) + self.eps
        expected_magnitude = torch.sqrt(expected_magnitude)

        # Normalizar
        normalized_x = x / expected_magnitude

        return normalized_x
