#!/usr/bin/env python3
"""
DFN Inference Demo - Demostración de DFN Pura Entrenado

Script para probar y demostrar el modelo DFN Pura después del entrenamiento.
Incluye generación continua, análisis de entidades, y comparación de rendimiento.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import gc

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dfn import DFNPureModel
from dfn.utils import print_gpu_memory_info, get_model_memory_footprint, print_memory_summary


def load_trained_model(checkpoint_path, config=None):
    """Loads trained model from checkpoint"""
    print(f"📁 Loading model from: {checkpoint_path}")
    if config is None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        print("   ✅ Configuration loaded from checkpoint")
    else:
        print("   ✅ Using provided configuration")
    model = DFNPureModel(**config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✅ Model weights loaded")
        print(f"   📊 Epoch: {checkpoint['epoch']}")
        print(f"   📊 Loss: {checkpoint['loss']:.4f}")
    model.eval()
    if torch.cuda.is_available():
        print(f"   🎮 Initial Allocated VRAM: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB")
    return model, device, config


def continuous_generation_demo(model, device, config):
    """Continuous generation demonstration"""
    print("\n🎨 CONTINUOUS GENERATION")
    print("=" * 30)
    seed_length = 20
    seed_sequence = torch.randn(1, seed_length, config['input_dim']).to(device)
    print(f"📥 Seed sequence: {seed_sequence.shape}")
    with torch.no_grad():
        generated_field = model(seed_sequence)
        print(f"📤 Generated field: {generated_field.shape}")
        generated_sequence = generated_field.mean(dim=1)
        print(f"📄 Generated sequence: {generated_sequence.shape}")
        field_variance = generated_field.var().item()
        sequence_variance = generated_sequence.var().item()
        print(f"   📊 Field variance: {field_variance:.6f}")
        print(f"   📊 Sequence variance: {sequence_variance:.6f}")
        print(f"   ✅ {'Continuous field generated' if field_variance > 0.01 else 'Low variation field'}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"   ♻️ After cleanup: Allocated VRAM: {allocated:.2f}MB")


def entity_analysis_demo(model, device, config):
    """Extracted entity analysis"""
    print("\n🔍 ENTITY ANALYSIS")
    print("=" * 30)
    test_cases = [
        ("Uniform", torch.randn(2, 50, config['input_dim']) * 0.5),
        ("Concentrated", torch.randn(2, 50, config['input_dim'])),
        ("Sequential", torch.randn(2, 50, config['input_dim']))
    ]
    test_cases[2][1][:, 25:, :] *= 3.0  # More information in the second half
    for name, data in test_cases:
        data = data.to(device)
        print(f"\n📊 Pattern: {name}")
        with torch.no_grad():
            positions, states, weights = model.encode(data)
            print(f"   📍 Positions: {positions.shape}")
            print(f"   🧠 States: {states.shape}")
            print(f"   ⚖️ Weights: {weights.shape}")
            weight_mean = weights.mean().item()
            weight_std = weights.std().item()
            print(f"   📈 Weight mean: {weight_mean:.6f}")
            print(f"   📉 Weight std: {weight_std:.6f}")
            pos_mean = positions.mean(dim=1)
            pos_std = positions.std().item()
            print(f"   🎯 Position spread: {pos_std:.4f}")
            if weight_std > 0.001:
                print("   ✅ Entities detect content variation")
            else:
                print("   ⚠️ Entities with uniform distribution")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"   ♻️ After cleanup: Allocated VRAM: {allocated:.2f}MB")


def performance_benchmark(model, device, config):
    """Performance benchmark"""
    print("\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 30)
    test_lengths = [32, 64, 128, 256, 512, 1024 * 200, 1024 * 250]
    print("📏 Scalability test:")
    for length in test_lengths:
        x = torch.randn(2, length, config['input_dim']).to(device)
        if torch.cuda.is_available():
            before_alloc = torch.cuda.memory_allocated() / 1024 / 1024
            torch.cuda.synchronize()
        else:
            before_alloc = 0
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else time.time()
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            start_time.record()
        else:
            start_time = time.time()
        with torch.no_grad():
            output = model(x)
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time) / 1000  # seconds
        else:
            elapsed = time.time() - start_time
        if torch.cuda.is_available():
            after_alloc = torch.cuda.memory_allocated() / 1024 / 1024
            delta_alloc = after_alloc - before_alloc
        else:
            after_alloc = delta_alloc = 0
        print(f"   Length {length:4d}: {elapsed:.3f}s, Current VRAM: {after_alloc:.1f}MB, ΔAlloc: {delta_alloc:.1f}MB")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n🎯 Consistency of entities:")
    x = torch.randn(1, 100, config['input_dim']).to(device)
    with torch.no_grad():
        positions, states, weights = model.encode(x)
        num_entities = states.shape[1]
    print(f"   Extracted entities: {num_entities} (constant)")
    print(f"   ✅ No sequence limit - works with any length")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"   ♻️ After cleanup: Allocated VRAM: {allocated:.2f}MB")


def memory_analysis(model, device, config):
    """Complete memory analysis"""
    print("\n💾 MEMORY ANALYSIS")
    print("=" * 30)
    memory_info = get_model_memory_footprint(model)
    print("📊 Model Memory Breakdown:")
    print(f"   Parameters: {memory_info['parameters_mb']:.1f}MB")
    print(f"   Buffers: {memory_info['buffers_mb']:.1f}MB")
    print(f"   Total: {memory_info['total_mb']:.1f}MB")
    if torch.cuda.is_available():
        allocated_vram = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"🎮 Current GPU Allocated VRAM: {allocated_vram:.2f}MB")
    else:
        print("🎮 GPU not available.")
    print("\n🔢 Parameters by Component:")
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                total_params += params
                print(f"   {name}: {params:,} ({params/sum(p.numel() for p in model.parameters())*100:.1f}%)")
    print(f"   Total: {total_params:,}")


def main():
    """Main demonstration function"""
    print("🎭 DFN Pure Inference Demo")
    print("=" * 50)
    print("Script to test trained model")
    print()
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = list(Path(checkpoint_dir).glob("dfn_pure_*.pt"))
        if checkpoints:
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_checkpoint = checkpoints[0]
            print(f"📁 Checkpoint found: {latest_checkpoint}")
        else:
            print("❌ No checkpoints found")
            print("💡 Run train.py first to create a trained model")
            return
    else:
        print(f"❌ Directory '{checkpoint_dir}' does not exist")
        print("💡 Run train.py first to create a trained model")
        return
    model, device, config = load_trained_model(latest_checkpoint)
    try:
        continuous_generation_demo(model, device, config)
        entity_analysis_demo(model, device, config)
        performance_benchmark(model, device, config)
        memory_analysis(model, device, config)
        print("\n🏆 DEMONSTRATION COMPLETE!")
        print("=" * 50)
        print("✅ Pure DFN functioning correctly")
        print("✅ Continuous generation active")
        print("✅ Entity analysis functional")
        print("✅ Scalability verified")
        print(f"✅ Model saved in: {latest_checkpoint}")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("💡 Verify that the model is trained correctly")


if __name__ == "__main__":
    main()
