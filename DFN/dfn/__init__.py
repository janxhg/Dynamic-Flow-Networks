"""
Dynamic Flow Networks (DFN) - Una Nueva Arquitectura Basada en Flujo Continuo

Esta implementación recrea la arquitectura DFN descrita en el paper:
"Dynamic Flow Networks (DFN): Una Nueva Arquitectura Basada en Flujo Continuo para Modelos de Inteligencia Artificial"

Componentes principales:
1. ContinuousField: Mapea entradas discretas a un campo continuo F(x)
2. DynamicFlowAttention: Implementa atención basada en flujo vectorial
3. FlowNorm: Normalización para estabilizar magnitudes de flujo
4. DynamicFieldMLP: Transformación de entidades con contexto local
5. PersistentFieldMemory: Memoria a largo plazo entre pasos

Modelos disponibles:
- DFNPureModel: DFN Pura sin vocabulario ni restricciones de secuencia

Características:
- Complejidad subcuadrática O(n log n)
- Entidades continuas en lugar de tokens discretos
- Atención local adaptativa basada en flujo
- Memoria persistente para contexto a largo plazo
- Escalable a contextos largos y multimodales

Autor: Implementado basado en el paper conceptual de DFN
"""

__version__ = "0.1.0"
__author__ = "DFN Development Team"

from .core import (
    ContinuousField,
    DynamicFlowAttention,
    FlowNorm,
    DynamicFieldMLP,
    PersistentFieldMemory,
    DFNLayer
)

from .models import (
    DFNPureModel
)

__all__ = [
    # Core components
    "ContinuousField",
    "DynamicFlowAttention",
    "FlowNorm",
    "DynamicFieldMLP",
    "PersistentFieldMemory",
    "DFNLayer",

    # Models
    "DFNPureModel"
]
