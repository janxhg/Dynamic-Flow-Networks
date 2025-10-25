"""
Pure DFN Model - Sin Vocabulario ni Secuencia Máxima

Este módulo implementa DFN siguiendo estrictamente el concepto del paper:
- Sin tokens discretos ni embeddings
- Sin posiciones fijas
- Campos continuos de entrada y salida
- Entidades adaptativas basadas en densidad de información
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..core import DFNLayer, FlowNorm


class DensityField(nn.Module):
    """Campo de densidad para entrada/salida continua"""

    def __init__(self, input_dim: int, field_dim: int, num_samples: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.field_dim = field_dim
        self.num_samples = num_samples

        # Proyección de entrada a campo denso
        self.density_projection = nn.Sequential(
            nn.Linear(input_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.GELU()
        )

        # Generador de coordenadas espaciales
        self.coordinate_generator = nn.Linear(field_dim, 2)  # [x, y] o [t, x] dependiendo del dominio

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convierte entrada discreta en campo de densidad continuo
        Args:
            x: [batch_size, seq_len, input_dim] entrada discreta
        Returns:
            coordinates: [batch_size, num_samples, 2] coordenadas espaciales
            densities: [batch_size, num_samples, field_dim] densidades en cada punto
        """
        batch_size, seq_len, _ = x.shape

        # Proyectar a espacio de densidad
        density_features = self.density_projection(x)  # [batch, seq_len, field_dim]

        # Generar coordenadas espaciales basadas en contenido
        # density_features: [batch, seq_len, field_dim] -> [batch * seq_len, field_dim]
        batch_size, seq_len, field_dim = density_features.shape
        density_flat = density_features.view(-1, field_dim)  # [batch * seq_len, field_dim]

        coords_flat = self.coordinate_generator(density_flat)  # [batch * seq_len, 2]
        coords = coords_flat.view(batch_size, seq_len, 2)  # [batch, seq_len, 2]

        # Crear campo continuo interpolando entre puntos discretos
        # Usar todos los puntos de la secuencia como muestras del campo
        coordinates = coords
        densities = density_features

        return coordinates, densities

    def sample_field(self, coordinates: torch.Tensor, densities: torch.Tensor,
                    num_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Muestrea el campo continuo en puntos adicionales"""
        if num_samples is None:
            num_samples = self.num_samples

        batch_size = coordinates.shape[0]

        # Generar puntos de muestreo adicionales basados en densidad
        # Más muestras donde la densidad es alta
        density_magnitude = torch.norm(densities, dim=-1)  # [batch, seq_len]

        # Normalizar para obtener pesos de muestreo
        weights = F.softmax(density_magnitude, dim=-1)

        # Muestreo ponderado de coordenadas existentes
        sampled_indices = torch.multinomial(weights, num_samples, replacement=True)

        # Obtener muestras
        sampled_coords = torch.gather(
            coordinates, 1,
            sampled_indices.unsqueeze(-1).expand(-1, -1, coordinates.shape[-1])
        )
        sampled_densities = torch.gather(
            densities, 1,
            sampled_indices.unsqueeze(-1).expand(-1, -1, densities.shape[-1])
        )

        return sampled_coords, sampled_densities


class ContinuousEntityExtractor(nn.Module):
    """Extrae entidades continuas sin embeddings discretos"""

    def __init__(self, field_dim: int, entity_dim: int, num_entities: int, coord_dim: int = 2):
        super().__init__()
        self.field_dim = field_dim
        self.entity_dim = entity_dim
        self.num_entities = num_entities
        self.coord_dim = coord_dim

        # Mapeo de densidad a entidades
        self.entity_mapper = nn.Sequential(
            nn.Linear(field_dim + coord_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, entity_dim + coord_dim + 1)  # estado, posición, peso
        )

    def forward(self, coordinates: torch.Tensor, densities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extrae entidades del campo continuo
        Args:
            coordinates: [batch_size, num_points, coord_dim] coordenadas
            densities: [batch_size, num_points, field_dim] densidades
        Returns:
            positions: [batch_size, num_entities, coord_dim] posiciones de entidades
            states: [batch_size, num_entities, entity_dim] estados de entidades
            weights: [batch_size, num_entities, 1] pesos de entidades
        """
        batch_size, num_points, _ = coordinates.shape

        # Concatenar coordenadas y densidades
        field_input = torch.cat([coordinates, densities], dim=-1)  # [batch, num_points, coord_dim + field_dim]

        # Mapear a entidades - procesar cada punto individualmente
        batch_size, num_points, _ = field_input.shape

        # field_input: [batch, num_points, coord_dim + field_dim] -> [batch * num_points, coord_dim + field_dim]
        field_flat = field_input.view(-1, field_input.shape[-1])  # [batch * num_points, coord_dim + field_dim]

        entity_features_flat = self.entity_mapper(field_flat)  # [batch * num_points, entity_dim + coord_dim + 1]
        entity_features = entity_features_flat.view(batch_size, num_points, -1)  # [batch, num_points, entity_dim + coord_dim + 1]

        # Calcular importancia basada en magnitud
        importance = torch.norm(entity_features, dim=-1)  # [batch, num_points]

        # Seleccionar entidades más importantes
        _, top_indices = torch.topk(importance, min(self.num_entities, num_points), dim=-1)

        positions = []
        states = []
        weights = []

        for b in range(batch_size):
            selected_features = entity_features[b][top_indices[b]]  # [num_entities, entity_dim + coord_dim + 1]

            # Separar componentes
            pos = selected_features[:, :self.coord_dim]
            state = selected_features[:, self.coord_dim:self.coord_dim + self.entity_dim]
            weight = selected_features[:, -1:]

            positions.append(pos.unsqueeze(0))
            states.append(state.unsqueeze(0))
            weights.append(weight.unsqueeze(0))

        positions = torch.cat(positions, dim=0)  # [batch, num_entities, coord_dim]
        states = torch.cat(states, dim=0)       # [batch, num_entities, entity_dim]
        weights = torch.cat(weights, dim=0)     # [batch, num_entities, 1]

        return positions, states, weights


class ContinuousOutputGenerator(nn.Module):
    """Genera salida continua sin vocabulario discreto"""

    def __init__(self, entity_dim: int, output_dim: int, field_resolution: int = 100):
        super().__init__()
        self.entity_dim = entity_dim
        self.output_dim = output_dim
        self.field_resolution = field_resolution

        # Proyección de entidades a densidad de salida
        self.output_projection = nn.Sequential(
            nn.Linear(entity_dim, entity_dim * 2),
            nn.GELU(),
            nn.Linear(entity_dim * 2, output_dim)
        )

        # Generador de campo de salida
        self.field_generator = nn.Linear(output_dim, field_resolution * 2)  # coordenadas + densidad

    def forward(self, entity_states: torch.Tensor, entity_positions: torch.Tensor) -> torch.Tensor:
        """Genera campo de salida continuo
        Args:
            entity_states: [batch_size, num_entities, entity_dim] estados de entidades
            entity_positions: [batch_size, num_entities, 2] posiciones de entidades
        Returns:
            output_field: [batch_size, field_resolution, output_dim] campo de salida
        """
        batch_size, num_entities, _ = entity_states.shape

        # Proyectar estados a espacio de salida
        projected_states = self.output_projection(entity_states)  # [batch, num_entities, output_dim]

        # Usar posiciones de entidades para generar campo de salida
        # Crear grid de coordenadas de salida
        grid_size = int(self.field_resolution ** 0.5)
        grid_coords = torch.linspace(-1, 1, grid_size, device=entity_states.device)
        grid_x, grid_y = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
        output_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [field_resolution, 2]

        # Expandir para cada batch
        output_coords = output_coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, field_resolution, 2]

        # Calcular densidad en cada punto del grid usando atención gaussiana
        output_field = []
        for b in range(batch_size):
            # Distancias desde entidades a puntos del grid
            distances = torch.cdist(output_coords[b], entity_positions[b])  # [field_resolution, num_entities]

            # Atención gaussiana basada en distancia
            attention_weights = torch.exp(-distances ** 2 / 0.1)  # sigma = 0.1

            # Normalizar atención
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Generar densidad en cada punto
            point_density = torch.matmul(attention_weights, projected_states[b])  # [field_resolution, output_dim]
            output_field.append(point_density.unsqueeze(0))

        output_field = torch.cat(output_field, dim=0)  # [batch, field_resolution, output_dim]

        return output_field


class DFNPureModel(nn.Module):
    """DFN Pura - Sin vocabulario ni secuencia máxima"""

    def __init__(self,
                 input_dim: int,           # Dimensión de entrada (ej: 768 para features de texto)
                 entity_dim: int = 256,    # Dimensión de estados de entidades
                 field_dim: int = 128,     # Dimensión del campo denso
                 num_entities: int = 64,   # Número de entidades a extraer
                 num_layers: int = 4,      # Número de capas DFN
                 coord_dim: int = 2,       # Dimensión de coordenadas (2D espacial)
                 k: int = 8,               # Vecinos para atención de flujo
                 alpha: float = 0.1,       # Fuerza del flujo
                 sigma: float = 1.0,       # Sigma para afinidad gaussiana
                 memory_size: int = 32,    # Tamaño de memoria persistente
                 beta: float = 0.9,        # Factor de momentum para memoria
                 dropout: float = 0.1,
                 field_resolution: int = 100):
        super().__init__()

        self.input_dim = input_dim
        self.entity_dim = entity_dim
        self.field_dim = field_dim
        self.num_entities = num_entities
        self.coord_dim = coord_dim

        # Campo de densidad de entrada
        self.density_field = DensityField(input_dim, field_dim, num_samples=1000)

        # Extractor de entidades continuas
        self.entity_extractor = ContinuousEntityExtractor(field_dim, entity_dim, num_entities, coord_dim)

        # Capas DFN
        self.dfn_layers = nn.ModuleList([
            DFNLayer(entity_dim, coord_dim, k, alpha, sigma, memory_size, beta, dropout)
            for _ in range(num_layers)
        ])

        # Generador de salida continua
        self.output_generator = ContinuousOutputGenerator(entity_dim, input_dim, field_resolution)

        # Inicialización
        self.apply(self._init_parameters)

    def _init_parameters(self, module):
        """Inicialización de parámetros"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_entities: bool = False) -> torch.Tensor:
        """Forward pass sin restricciones de vocabulario o secuencia
        Args:
            x: [batch_size, seq_len, input_dim] entrada de cualquier longitud
            return_entities: si retornar entidades intermedias
        Returns:
            output_field: [batch_size, field_resolution, input_dim] campo de salida continuo
            entities: entidades procesadas (opcional)
        """
        batch_size, seq_len, input_dim = x.shape

        # 1. Convertir entrada a campo de densidad
        coordinates, densities = self.density_field(x)

        # 2. Extraer entidades continuas
        entity_positions, entity_states, entity_weights = self.entity_extractor(coordinates, densities)

        # 3. Procesar con capas DFN
        current_states = entity_states
        current_positions = entity_positions

        for layer in self.dfn_layers:
            current_states, current_positions = layer(current_states, current_positions)

        # 4. Generar salida continua
        output_field = self.output_generator(current_states, current_positions)

        if return_entities:
            return output_field, (current_states, current_positions, entity_weights)

        return output_field

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solo encoding - retorna entidades sin procesar"""
        batch_size, seq_len, _ = x.shape

        # Campo de densidad
        coordinates, densities = self.density_field(x)

        # Extraer entidades
        positions, states, weights = self.entity_extractor(coordinates, densities)

        return positions, states, weights

    def decode(self, entity_states: torch.Tensor, entity_positions: torch.Tensor) -> torch.Tensor:
        """Solo decoding - genera salida desde entidades"""
        return self.output_generator(entity_states, entity_positions)

    def reset_memory(self):
        """Reinicia memoria de todas las capas"""
        for layer in self.dfn_layers:
            layer.reset_memory()
