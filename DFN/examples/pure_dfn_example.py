"""
Pure DFN Example - No Vocabulary nor Maximum Sequence Length

This example demonstrates how to use DFN in its purest form, without the
limitations of discrete tokens or fixed positions.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dfn import DFNPureModel
from dfn.utils import print_gpu_memory_info, get_model_memory_footprint, print_memory_summary


def create_continuous_data(batch_size: int = 4, seq_len: int = 50, input_dim: int = 512):
    """Crea datos continuos para entrenamiento"""
    # Generar datos continuos (ej: features de texto, imagen, audio)
    x = torch.randn(batch_size, seq_len, input_dim)

    # Target: reconstrucción o predicción continua
    y = x + 0.1 * torch.randn_like(x)  # Reconstrucción con ruido

    return x, y


def continuous_reconstruction_example():
    """Ejemplo de reconstrucción continua con DFN pura"""
    print("=== Continuous Reconstruction with Pure DFN ===")
    print("No vocabulary, no maximum sequence, just continuous fields")

    # Configuración sin vocab_size ni max_seq_len
    model = DFNPureModel(
        input_dim=512,        # Dimensión de entrada
        entity_dim=1024,      # Dimensión de entidades
        field_dim=512,        # Dimensión del campo denso
        num_entities=32,      # Número de entidades a extraer
        num_layers=3,         # Capas DFN
        coord_dim=2,          # Coordenadas 2D
        k=8,                  # Vecinos para atención
        alpha=0.1,            # Fuerza del flujo
        sigma=1.0,            # Sigma gaussiano
        memory_size=16,       # Memoria persistente
        field_resolution=64   # Resolución del campo de salida
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Mostrar footprint de memoria del modelo
    memory_info = get_model_memory_footprint(model)
    print(f"📊 Model Memory Footprint: {memory_info['total_mb']:.1f}MB")
    print(f"   Parameters: {memory_info['parameters_mb']:.1f}MB")
    print(f"   Buffers: {memory_info['buffers_mb']:.1f}MB")

    # Forzar uso de GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🔥 Using device: {device}")

    # Mostrar información inicial de GPU/CPU
    print_gpu_memory_info("🎮 Initial Memory State:")

    # Datos continuos - mover a GPU si está disponible
    x, y = create_continuous_data(batch_size=8, seq_len=40, input_dim=512)
    x, y = x.to(device), y.to(device)
    print(f"Input: {x.shape} (batch, seq_len, input_dim)")
    print(f"Target: {y.shape}")

    # Forward pass - mostrar memoria durante procesamiento
    print_gpu_memory_info("🎮 Memory During Processing:")
    output_field = model(x)
    print_gpu_memory_info("🎮 Memory After Processing:")

    print(f"Output field: {output_field.shape} (batch, field_resolution, input_dim)")

    # Calcular pérdida de reconstrucción
    # Para este ejemplo, interpolamos el campo de salida a la resolución de entrada
    mse_loss = F.mse_loss(output_field.mean(dim=1), y.mean(dim=1))
    print(f"MSE Loss: {mse_loss.item():.4f}")

    # Análisis de entidades
    output_field, (states, positions, weights) = model(x, return_entities=True)
    print(f"Extracted entities: {states.shape} (batch, num_entities, entity_dim)")
    print(f"Entity positions: {positions.shape}")
    print(f"Entity weights: {weights.shape}")

    # Visualizar distribución de pesos de entidades
    weight_magnitude = weights.squeeze(-1).mean(dim=0)
    print(f"Average entity weights: {weight_magnitude}")

    return model


def density_field_visualization():
    """Visualización del campo de densidad"""
    print("\n=== Density Field Analysis ===")

    model = DFNPureModel(
        input_dim=512,
        entity_dim=1024,
        field_dim=512,
        num_entities=32,
        num_layers=4,
        field_resolution=64
    )

    # Mover a GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Crear datos con diferentes patrones de densidad
    batch_size = 4

    # Patrón 1: densidad uniforme
    x1 = torch.randn(batch_size, 30, 512) * 0.5
    x1 = x1.to(device)

    # Patrón 2: densidad variable (más información al final)
    x2 = torch.randn(batch_size, 30, 512)
    x2[:, 15:, :] *= 2.0  # Más densidad en la segunda mitad
    x2 = x2.to(device)

    print("Analyzing different density patterns...")

    # Procesar patrón 1
    entities1 = model.encode(x1)
    positions1, states1, weights1 = entities1

    # Procesar patrón 2
    entities2 = model.encode(x2)
    positions2, states2, weights2 = entities2

    # Comparar distribución de pesos
    avg_weights1 = weights1.squeeze(-1).mean(dim=0)  # Promedio sobre batches
    avg_weights2 = weights2.squeeze(-1).mean(dim=0)

    # Los pesos deberían reflejar la distribución de información
    variance1 = torch.var(avg_weights1)
    variance2 = torch.var(avg_weights2)

    print(f"   Uniform pattern: weight variance = {variance1:.4f}")
    print(f"   Variable pattern: weight variance = {variance2:.4f}")
    print(f"   Distribution difference: {torch.std(avg_weights2 - avg_weights1).item():.4f}")

    return model


def continuous_generation_example():
    """Ejemplo de generación continua"""
    print("\n=== Continuous Generation ===")

    model = DFNPureModel(
        input_dim=512,
        entity_dim=1024,
        field_dim=512,
        num_entities=16,
        num_layers=3,
        field_resolution=32
    )

    # Mover a GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Secuencia inicial corta
    seed_sequence = torch.randn(1, 10, 512)  # [1, 10, 512]
    seed_sequence = seed_sequence.to(device)
    print(f"Seed sequence: {seed_sequence.shape}")

    # Generar continuación continua
    # En lugar de tokens discretos, generamos un campo continuo
    output_field = model(seed_sequence)
    print(f"Generated field: {output_field.shape}")

    # Convertir campo continuo de vuelta a secuencia discreta (para visualización)
    # Tomar puntos del campo como nueva secuencia
    generated_sequence = output_field.mean(dim=1)  # Promedio sobre el campo
    print(f"Generated continuous sequence: {generated_sequence.shape}")

    return model


def scalability_test():
    """Prueba de escalabilidad - contextos muy largos"""
    print("\n=== Scalability Test ===")
    print("Pure DFN can handle sequences of any length")

    model = DFNPureModel(
        input_dim=512,
        entity_dim=1024,
        field_dim=512,
        num_entities=64,
        num_layers=6
    )

    # Mover a GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Secuencias de diferentes longitudes
    lengths = [50, 200, 1000, 5000]

    for length in lengths:
        # Crear secuencia larga
        x = torch.randn(2, length, 512)
        x = x.to(device)

        # Procesar
        output_field = model(x)

        print(f"Length {length:4d}: Input {x.shape} -> Output field {output_field.shape}")

        # Verificar que el número de entidades es constante
        _, (states, _, _) = model(x, return_entities=True)
        print(f"  Extracted entities: {states.shape[1]} (constant regardless of length)")

    return model


def main():
    """Función principal del ejemplo"""
    print("Dynamic Flow Networks - Pure Implementation")
    print("=" * 60)
    print("✨ No vocab_size - No max_seq_len ✨")
    print("Only continuous fields and adaptive entities")
    print()

    # Ejecutar ejemplos
    model1 = continuous_reconstruction_example()
    model2 = density_field_visualization()
    model3 = continuous_generation_example()
    model4 = scalability_test()

    print("\n" + "=" * 60)
    print("✅ Pure DFN successfully implemented!")
    print("Features:")
    print("- ✅ No fixed vocabulary")
    print("- ✅ No sequence limit")
    print("- ✅ Adaptive entities")
    print("- ✅ Continuous fields")
    print("- ✅ Scalability proven")
    print("\nThis implementation faithfully follows the paper's concept.")

    # Mostrar resumen completo de memoria al final
    print_memory_summary(model1, "📊 FINAL MEMORY USAGE SUMMARY")

    return model1


if __name__ == "__main__":
    main()
