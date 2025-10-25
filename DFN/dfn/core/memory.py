"""
Persistent Memory Module for Dynamic Flow Networks

This module implements PersistentFieldMemory for maintaining long-term
context across processing steps in the DFN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PersistentFieldMemory(nn.Module):
    """Memoria persistente de campo para contexto a largo plazo"""

    def __init__(self, dim: int, memory_size: int = 64, beta: float = 0.9):
        super().__init__()
        self.dim = dim
        self.memory_size = memory_size
        self.beta = beta

        # Memoria persistente inicializada en cero
        self.register_buffer('memory', torch.zeros(memory_size, dim))

        # Proyección para agregar nueva información a memoria
        self.memory_projection = nn.Linear(dim, dim)
        self.aggregate_projection = nn.Linear(dim, dim)

    def forward(self, states: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Actualiza y consulta la memoria persistente
        Args:
            states: [batch_size, num_entities, dim] estados actuales
            positions: [batch_size, num_entities, pos_dim] posiciones (opcional)
        Returns:
            memory_context: [batch_size, num_entities, dim] contexto de memoria
        """
        batch_size, num_entities, dim = states.shape

        # Agregar información global de todas las entidades
        global_state = torch.mean(states, dim=1)  # [batch_size, dim]
        aggregated_info = self.aggregate_projection(global_state)  # [batch_size, dim]

        # Actualizar memoria con momentum
        # Expandir memoria para cada batch
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, memory_size, dim]

        # Proyectar información agregada
        update_info = self.memory_projection(aggregated_info)  # [batch, dim]

        # Actualización con momentum
        updated_memory = self.beta * self.memory + (1 - self.beta) * update_info.mean(dim=0)

        # Actualizar memoria del módulo (promedio global)
        self.memory.data = updated_memory

        # Proporcionar contexto de memoria a todas las entidades
        # Usar atención basada en similitud coseno con estados actuales
        memory_context = self.memory_projection(self.memory)  # [memory_size, dim]

        # Calcular similitud entre estados actuales y memoria
        states_flat = states.view(-1, dim)  # [batch*num_entities, dim]
        memory_context_expanded = memory_context.unsqueeze(0).expand(len(states_flat), -1, -1)
        memory_context_expanded = memory_context_expanded.view(len(states_flat), self.memory_size, dim)

        # Similitud coseno
        similarity = F.cosine_similarity(
            states_flat.unsqueeze(1),
            memory_context_expanded,
            dim=-1
        )  # [batch*num_entities, memory_size]

        # Atención softmax sobre memoria
        attention_weights = F.softmax(similarity, dim=-1)  # [batch*num_entities, memory_size]

        # Agregar contexto de memoria ponderado
        context_expanded = memory_context.unsqueeze(0).expand(len(states_flat), -1, -1)
        weighted_context = attention_weights.unsqueeze(-1) * context_expanded  # [batch*num_entities, memory_size, dim]
        memory_output = weighted_context.sum(dim=1)  # [batch*num_entities, dim]

        # Reshape de vuelta
        memory_context = memory_output.view(batch_size, num_entities, dim)

        return memory_context

    def reset_memory(self):
        """Reinicia la memoria a cero"""
        self.memory.data.zero_()
