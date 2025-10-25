#!/usr/bin/env python3
"""
DFN Pure Inference Script - Complete Inference for Pure DFN Models

Full script to perform inference with Pure DFN models (.pt) featuring:
- Detailed RAM/VRAM monitoring
- Support for different input types
- Continuous text generation
- Entity and field analysis
- Multiple inference modes

Usage:
    python inference.py --model_path checkpoints/model.pt --input_type text --text "Hello world"
    python inference.py --model_path checkpoints/model.pt --input_type embeddings --batch_size 4
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
import argparse
import json
import time
from pathlib import Path
import psutil
import gc

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dfn.models.dfn_pure import DFNPureModel
from dfn.utils import print_gpu_memory_info, get_model_memory_footprint, print_memory_summary


# =============================================================================
# 📊 MONITOREO DE MEMORIA Y RENDIMIENTO
# =============================================================================

class MemoryProfiler:
    """Full profiler for RAM/VRAM monitoring"""

    def __init__(self):
        self.start_time = time.time()
        self.snapshots = []

    def take_snapshot(self, label: str = ""):
        """Take a snapshot of current memory state"""
        snapshot = {
            'label': label,
            'time_elapsed': time.time() - self.start_time,
            'ram': self._get_ram_info(),
            'gpu': self._get_gpu_info() if torch.cuda.is_available() else None
        }
        self.snapshots.append(snapshot)
        return snapshot

    def _get_ram_info(self) -> Dict[str, float]:
        """Get system RAM information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent
        }

    def _get_gpu_info(self) -> Dict[str, float]:
        """Get GPU information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - allocated,
                'utilization_percent': (allocated / total) * 100 if total > 0 else 0
            }
        return {}

    def print_report(self):
        """Print complete memory report"""
        print("\n" + "="*60)
        print("📊 FULL MEMORY REPORT")
        print("="*60)

        for i, snapshot in enumerate(self.snapshots):
            print(f"\n📍 Snapshot {i+1}: {snapshot['label']}")
            print(f"   ⏱️  Time: {snapshot['time_elapsed']:.2f}s")

            # RAM info
            ram = snapshot['ram']
            print("   💾 RAM:")
            print(f"      Total: {ram['total_gb']:.2f}GB")
            print(f"      Used: {ram['used_gb']:.2f}GB ({ram['usage_percent']:.1f}%)")
            print(f"      Available: {ram['available_gb']:.2f}GB")

            # GPU info
            if snapshot['gpu']:
                gpu = snapshot['gpu']
                print("   🎮 VRAM:")
                print(f"      Total: {gpu['total_gb']:.2f}GB")
                print(f"      Allocated: {gpu['allocated_gb']:.2f}GB ({gpu['utilization_percent']:.1f}%)")
                print(f"      Reserved: {gpu['reserved_gb']:.2f}GB")
                print(f"      Free: {gpu['free_gb']:.2f}GB")

    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory used"""
        if not self.snapshots:
            return {}

        peak_ram = max(s['ram']['used_gb'] for s in self.snapshots)
        peak_gpu = max((s['gpu']['allocated_gb'] for s in self.snapshots if s['gpu']), default=0)

        return {
            'peak_ram_gb': peak_ram,
            'peak_vram_gb': peak_gpu
        }


# =============================================================================
# 🔧 CONFIGURACIÓN Y CARGA DE MODELOS
# =============================================================================

class InferenceConfig:
    """Inference configuration"""

    def __init__(self,
                 model_path: str,
                 device: Optional[str] = None,
                 precision: str = "fp32",
                 max_batch_size: int = 8,
                 field_resolution: int = 100,
                 memory_efficient: bool = True):
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = precision  # fp32, fp16, bf16
        self.max_batch_size = max_batch_size
        self.field_resolution = field_resolution
        self.memory_efficient = memory_efficient

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'device': self.device,
            'precision': self.precision,
            'max_batch_size': self.max_batch_size,
            'field_resolution': self.field_resolution,
            'memory_efficient': self.memory_efficient
        }


class ModelLoader:
    """Model loader with memory optimization"""

    @staticmethod
    def load_model(config: InferenceConfig) -> Tuple[DFNPureModel, Dict[str, Any]]:
        """Load model from checkpoint"""
        print(f"🔄 Loading model from: {config.model_path}")

        # Verificar que el archivo existe
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model not found: {config.model_path}")

        # Cargar checkpoint
        profiler = MemoryProfiler()
        profiler.take_snapshot("Start loading")

        print("   💾 Reading checkpoint...")
        checkpoint = torch.load(config.model_path, map_location='cpu')

        # Extraer configuración del modelo
        model_config = checkpoint.get('config', {})
        print(f"   ⚙️  Model config: {model_config}")

        # Crear modelo
        print("   🏗️  Creating model...")
        model = DFNPureModel(**model_config)

        # Configurar precisión
        if config.precision == "fp16" and torch.cuda.is_available():
            model = model.half()
            print("   🎯 Precision: FP16")
        elif config.precision == "bf16" and torch.cuda.is_available():
            model = model.to(torch.bfloat16)
            print("   🎯 Precision: BF16")
        else:
            print("   🎯 Precision: FP32")

        # Cargar pesos
        print("   📦 Loading weights...")
        model.load_state_dict(checkpoint['model_state_dict'])

        # Mover a dispositivo
        model = model.to(config.device)
        model.eval()

        profiler.take_snapshot("Model loaded")

        # Información del checkpoint
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'model_config': model_config,
            'memory_footprint': get_model_memory_footprint(model),
            'total_params': sum(p.numel() for p in model.parameters())
        }

        print(f"   ✅ Model loaded successfully!")
        print(f"   📊 Parameters: {info['total_params']:,}")
        print(f"   💾 Memory footprint: {info['memory_footprint']['total_mb']:.1f}MB")
        print(f"   🎯 Epoch: {info['epoch']}, Loss: {info['loss']:.4f}")

        # Limpiar memoria
        del checkpoint
        if config.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        profiler.take_snapshot("Cleanup completed")
        profiler.print_report()

        return model, info


# =============================================================================
# 🔮 FUNCIONES DE INFERENCIA
# =============================================================================

class DFNInference:
    """Inference engine for Pure DFN"""

    def __init__(self, model: DFNPureModel, config: InferenceConfig):
        self.model = model
        self.config = config
        self.profiler = MemoryProfiler()
        self.profiler.take_snapshot("Start inference")

    def encode_text(self, text_input: Union[str, List[str]],
                   tokenizer=None, max_length: int = 256) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encodes text into continuous embeddings"""
        self.profiler.take_snapshot("Start encoding")

        # Convertir texto a embeddings (simplificado)
        if isinstance(text_input, str):
            text_input = [text_input]

        # Crear embeddings sintéticos (reemplazar con tokenizer real si se tiene)
        embeddings_list = []
        for text in text_input:
            # Embedding simple basado en caracteres
            chars = list(text)
            embedding = self._text_to_embedding(chars, max_length, self.model.input_dim)
            embeddings_list.append(embedding)

        x = torch.stack(embeddings_list).to(self.config.device)

        # Encoding con el modelo
        with torch.no_grad():
            positions, states, weights = self.model.encode(x)

        self.profiler.take_snapshot("Encoding complete")

        info = {
            'input_shape': x.shape,
            'num_entities': states.shape[1],
            'entity_dim': states.shape[2],
            'positions': positions,
            'states': states,
            'weights': weights
        }

        return states, info

    def generate_continuous(self, entity_states: torch.Tensor,
                          entity_positions: torch.Tensor,
                          steps: int = 50) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generates a continuous field from entities"""
        self.profiler.take_snapshot("Start generation")

        # Generación paso a paso
        current_states = entity_states
        current_positions = entity_positions
        generated_sequence = []

        with torch.no_grad():
            for step in range(steps):
                # Aplicar una capa DFN para evolución temporal
                if step < len(self.model.dfn_layers):
                    layer = self.model.dfn_layers[step]
                    current_states, current_positions = layer(current_states, current_positions)

                # Generar campo de salida
                output_field = self.model.decode(current_states, current_positions)
                generated_sequence.append(output_field)

                # Evolución simple de entidades (para generación)
                # Añadir variación gaussiana pequeña
                noise = torch.randn_like(current_states) * 0.01
                current_states = current_states + noise

        # Combinar secuencia generada
        if generated_sequence:
            final_output = torch.stack(generated_sequence)
        else:
            final_output = self.model.decode(current_states, current_positions)

        self.profiler.take_snapshot("Generation complete")

        info = {
            'generated_shape': final_output.shape,
            'steps': steps,
            'final_positions': current_positions,
            'final_states': current_states
        }

        return final_output, info

    def full_inference(self, text_input: Union[str, List[str]],
                      generation_steps: int = 10) -> Dict[str, Any]:
        """Complete inference: encoding + generation"""
        self.profiler.take_snapshot("Start full inference")

        # 1. Encoding
        entity_states, encode_info = self.encode_text(text_input)

        # 2. Generación continua
        entity_positions = encode_info['positions']
        generated_output, gen_info = self.generate_continuous(
            entity_states, entity_positions, generation_steps
        )

        # 3. Análisis de entidades
        entity_analysis = self._analyze_entities(
            encode_info['positions'],
            encode_info['states'],
            encode_info['weights']
        )

        self.profiler.take_snapshot("Inference completed")

        results = {
            'input_text': text_input if isinstance(text_input, str) else text_input[0],
            'encoding_info': {
                'input_shape': encode_info['input_shape'],
                'num_entities': encode_info['num_entities'],
                'entity_dim': encode_info['entity_dim'],
                'positions': encode_info['positions'].detach().cpu().numpy(),
                'states': encode_info['states'].detach().cpu().numpy(),
                'weights': encode_info['weights'].detach().cpu().numpy()
            },
            'generation_info': {
                'generated_shape': gen_info['generated_shape'],
                'steps': gen_info['steps'],
                'final_positions': gen_info['final_positions'].detach().cpu().numpy(),
                'final_states': gen_info['final_states'].detach().cpu().numpy()
            },
            'entity_analysis': entity_analysis,
            'generated_output': generated_output.detach().cpu().numpy(),
            'memory_report': self.profiler.snapshots[-1]
        }

        return results

    def _text_to_embedding(self, chars: List[str], max_length: int, input_dim: int = 256) -> torch.Tensor:
        """Convert text to embedding tensor (simplified)"""
        # Crear embedding simple basado en código ASCII con la dimensión correcta
        embeddings = []
        for char in chars[:max_length]:
            char_code = ord(char)
            # Crear vector de características de la dimensión correcta
            embedding = []

            # Componentes básicas
            embedding.append(char_code / 255.0)  # Código normalizado
            embedding.append(len(chars) / 100.0)  # Longitud relativa

            # Componentes trigonométricos
            for i in range(1, 6):
                embedding.append(np.sin(char_code / (10.0 * i)))  # Diferentes frecuencias
                embedding.append(np.cos(char_code / (10.0 * i)))

            # Componentes adicionales hasta llegar a input_dim
            while len(embedding) < input_dim:
                # Añadir variaciones basadas en posición del carácter
                pos_factor = len(embeddings) / max_length
                embedding.append(np.sin(pos_factor * 2 * np.pi))
                embedding.append(np.cos(pos_factor * 2 * np.pi))

                # Añadir ruido controlado si aún necesitamos más dimensiones
                if len(embedding) < input_dim:
                    embedding.append(np.random.normal(0, 0.1))

            # Tomar solo las primeras input_dim dimensiones
            embedding = embedding[:input_dim]

            # Añadir algo de variabilidad
            embedding = np.array(embedding)
            noise = np.random.normal(0, 0.05, input_dim)
            embedding = embedding + noise

            embeddings.append(embedding)

        # Padding si es necesario
        while len(embeddings) < max_length:
            # Crear padding con ceros + ruido pequeño
            padding = np.random.normal(0, 0.01, input_dim)
            embeddings.append(padding)

        return torch.FloatTensor(np.array(embeddings))

    def _analyze_entities(self, positions: torch.Tensor, states: torch.Tensor,
                         weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze extracted entities"""
        batch_size, num_entities, coord_dim = positions.shape
        _, _, entity_dim = states.shape

        # Estadísticas de posiciones
        pos_mean = positions.mean(dim=1)
        pos_std = positions.std(dim=1)

        # Estadísticas de estados
        state_magnitude = torch.norm(states, dim=-1)
        state_mean = states.mean(dim=1)
        state_std = states.std(dim=1)

        # Análisis de pesos
        weights_sum = weights.sum(dim=1)

        analysis = {
            'num_entities': num_entities,
            'entity_dim': entity_dim,
            'coord_dim': coord_dim,
            'position_stats': {
                'mean': pos_mean.cpu().numpy(),
                'std': pos_std.cpu().numpy()
            },
            'state_stats': {
                'magnitude_mean': state_magnitude.mean(dim=1).cpu().numpy(),
                'magnitude_std': state_magnitude.std(dim=1).cpu().numpy(),
                'state_mean': state_mean.cpu().numpy(),
                'state_std': state_std.cpu().numpy()
            },
            'weights_stats': {
                'sum': weights_sum.cpu().numpy(),
                'max': weights.max(dim=1)[0].cpu().numpy(),
                'mean': weights.mean(dim=1).cpu().numpy()
            }
        }

        return analysis

    def benchmark_inference(self, input_text: str, num_runs: int = 10) -> Dict[str, Any]:
        """Performance benchmark"""
        print(f"\n🏃 Benchmark with {num_runs} runs...")

        times = []
        memory_usage = []

        for i in range(num_runs):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            start_time = time.time()

            with torch.no_grad():
                results = self.full_inference(input_text, generation_steps=5)

            end_time = time.time()
            times.append(end_time - start_time)

            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB

            print(f"   Run {i+1}/{num_runs}: {times[-1]:.3f}s")

        # Estadísticas
        times = np.array(times)
        memory_usage = np.array(memory_usage) if memory_usage else np.array([])

        benchmark_results = {
            'avg_time': times.mean(),
            'std_time': times.std(),
            'min_time': times.min(),
            'max_time': times.max(),
            'avg_memory_mb': memory_usage.mean() if len(memory_usage) > 0 else 0,
            'peak_memory_mb': memory_usage.max() if len(memory_usage) > 0 else 0,
            'throughput': 1.0 / times.mean()  # inferencias por segundo
        }

        print(f"\n📈 Benchmark results:")
        print(f"   Average time: {benchmark_results['avg_time']:.3f}s ± {benchmark_results['std_time']:.3f}s")
        print(f"   Throughput: {benchmark_results['throughput']:.2f} inf/s")
        if len(memory_usage) > 0:
            print(f"   Average memory: {benchmark_results['avg_memory_mb']:.1f}MB")
            print(f"   Peak memory: {benchmark_results['peak_memory_mb']:.1f}MB")

        return benchmark_results


# =============================================================================
# 🎯 FUNCIONES DE PROCESAMIENTO DE DATOS
# =============================================================================

def create_synthetic_embeddings(batch_size: int, seq_len: int, input_dim: int) -> torch.Tensor:
    """Create synthetic embeddings for testing"""
    # Generar datos tipo-texto estructurado
    data = []

    for sample in range(batch_size):
        sequence = []
        for pos in range(seq_len):
            # Simular diferentes tipos de tokens
            token_type = np.random.choice(['noun', 'verb', 'adj', 'func'], p=[0.4, 0.3, 0.2, 0.1])

            if token_type == 'noun':
                embedding = np.random.normal(0.2, 0.3, input_dim)
            elif token_type == 'verb':
                embedding = np.random.normal(0.1, 0.5, input_dim)
            elif token_type == 'adj':
                embedding = np.random.normal(-0.1, 0.4, input_dim)
            else:
                embedding = np.random.normal(0, 0.2, input_dim)

            # Coherencia posicional
            position_factor = np.sin(pos / seq_len * 2 * np.pi) * 0.1
            embedding += position_factor

            # Coherencia semántica
            if pos > 0:
                coherence = 0.3 * sequence[-1]
                embedding = 0.7 * embedding + coherence

            sequence.append(embedding)

        data.append(np.array(sequence))

    return torch.FloatTensor(np.array(data))


# =============================================================================
# 🎮 INTERFAZ PRINCIPAL
# =============================================================================

def run_inference_example(config: InferenceConfig, input_type: str = "text",
                         input_text: str = "Hello world DFN", batch_size: int = 1):
    """Run inference example"""

    print("\n" + "="*70)
    print("🚀 DFN PURE INFERENCE")
    print("="*70)

    # 1. Cargar modelo
    model, model_info = ModelLoader.load_model(config)

    # 2. Crear motor de inferencia
    inference_engine = DFNInference(model, config)

    # 3. Ejecutar inferencia según tipo
    if input_type == "text":
        print(f"\n📝 Input: {input_text}")
        results = inference_engine.full_inference(input_text, generation_steps=20)

        # Mostrar resultados
        print(f"\n📊 Extracted entities: {results['encoding_info']['num_entities']}")
        print(f"🔢 Entity dimension: {results['encoding_info']['entity_dim']}")
        print(f"📈 Output shape: {results['generation_info']['generated_shape']}")

        # Análisis de entidades
        entities = results['entity_analysis']
        print(f"\n🧠 ENTITY ANALYSIS:")
        print(f"   Positions - Mean: {entities['position_stats']['mean'].mean():.3f}")
        print(f"   States - Mean magnitude: {entities['state_stats']['magnitude_mean'].mean():.3f}")
        print(f"   Weights - Total sum: {entities['weights_stats']['sum'].mean():.3f}")

    elif input_type == "batch":
        print(f"\n📦 Processing batch of {batch_size} samples...")
        synthetic_data = create_synthetic_embeddings(batch_size, 64, model_info['model_config']['input_dim'])
        synthetic_data = synthetic_data.to(config.device)

        with torch.no_grad():
            output_field = model(synthetic_data)

        print(f"   ✅ Output shape: {output_field.shape}")
        print(f"   📊 Output range: [{output_field.min():.3f}, {output_field.max():.3f}]")

    # 4. Benchmark
    benchmark_results = inference_engine.benchmark_inference(input_text, num_runs=5)

    # 5. Reporte final
    print(f"\n🎯 PERFORMANCE:")
    print(f"   Throughput: {benchmark_results['throughput']:.2f} inf/s")
    print(f"   Latency: {benchmark_results['avg_time']*1000:.1f}ms")
    print(f"   VRAM: {benchmark_results['avg_memory_mb']:.1f}MB")
    save_results(results, benchmark_results, config)

    # Print raw model output for inspection
    print("\n🟢 Raw model output (results['generated_output']):")
    print(results['generated_output'])

    return results, benchmark_results


def save_results(results: Dict, benchmark: Dict, config: InferenceConfig):
    """Save inference results"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Convertir tensores a arrays numpy para JSON serializable
    def tensor_to_serializable(obj):
        """Convierte tensores y otros objetos no serializables"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_serializable(item) for item in obj]
        else:
            return obj

    # Convertir resultados a formato serializable
    serializable_results = tensor_to_serializable(results)

    output_data = {
        'timestamp': timestamp,
        'config': config.to_dict(),
        'benchmark': benchmark,
        'inference_results': serializable_results
    }

    output_file = f"inference_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved: {output_file}")


# =============================================================================
# 🎯 EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="DFN Pure Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the .pt model")
    parser.add_argument("--input_type", type=str, default="text",
                       choices=["text", "batch", "embeddings"],
                       help="Input type")
    parser.add_argument("--text", type=str, default="Hello world DFN",
                       help="Input text")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cpu, cuda)")
    parser.add_argument("--precision", type=str, default="fp32",
                       choices=["fp32", "fp16", "bf16"],
                       help="Numeric precision")
    parser.add_argument("--max_batch_size", type=int, default=8,
                       help="Maximum batch size")

    args = parser.parse_args()

    # Crear configuración
    config = InferenceConfig(
        model_path=args.model_path,
        device=args.device,
        precision=args.precision,
        max_batch_size=args.max_batch_size
    )

    # Ejecutar inferencia
    try:
        results, benchmark = run_inference_example(
            config, args.input_type, args.text, args.batch_size
        )
        print("\n✅ Inference completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Inference interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
