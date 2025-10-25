#!/usr/bin/env python3
"""
DFN Training Script - Entrenamiento de DFN Pura

Script completo para entrenar DFN Pura con monitoreo de VRAM y configuración simple.
Solo modifica la sección de hiperparámetros al principio.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time
import json

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dfn import DFNPureModel
from dfn.utils import print_gpu_memory_info, get_model_memory_footprint, print_memory_summary


# =============================================================================
# 🎛️  HIPERPARÁMETROS - MODIFICA SOLO ESTA SECCIÓN
# =============================================================================

class TrainingConfig:
    """Configuración de entrenamiento - modifica solo esto"""

    # === CONFIGURACIÓN DEL MODELO ===
    MODEL_CONFIG = {
        'input_dim': 256,         # Más pequeño para pruebas rápidas
        'entity_dim': 512,        # Más pequeño
        'field_dim': 256,         # Más pequeño
        'num_entities': 32,       # Menos entidades
        'num_layers': 3,          # Menos capas
        'coord_dim': 2,           # 2D
        'k': 8,                   # Menos vecinos
        'alpha': 0.1,
        'sigma': 1.0,
        'memory_size': 32,
        'beta': 0.9,
        'dropout': 0.1,
        'field_resolution': 64    # Más pequeño
    }

    # === CONFIGURACIÓN DE ENTRENAMIENTO ===
    TRAINING_CONFIG = {
        'batch_size': 16,         # Más grande para más rápido
        'seq_len': 256,            # Más corto para más rápido
        'num_epochs': 5,          # Menos épocas para pruebas
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'gradient_clip': 1.0,
        'validation_split': 0.1,
        'checkpoint_interval': 1, # Guardar cada época
        'output_dir': 'checkpoints'
    }

    # === CONFIGURACIÓN DE DATOS ===
    DATA_CONFIG = {
        'train_samples': 2000,    # Reducido para pruebas rápidas
        'val_samples': 500,       # Reducido para pruebas rápidas
        'noise_factor': 0.1,      # Ruido para reconstrucción
        'min_seq_len': 8,         # Mínimo para longitudes variables
        'max_seq_len': 64,        # Máximo para datos de entrenamiento
        'data_type': 'complex_text'  # Cambiar a complex_text que funciona
    }


# =============================================================================
# 🔧  FUNCIONES AUXILIARES
# =============================================================================

def create_training_data(config):
    """Crea datos de entrenamiento sintéticos"""
    print("📊 Generating training data...")

    train_size = config.DATA_CONFIG['train_samples']
    val_size = config.DATA_CONFIG['val_samples']
    batch_size = config.TRAINING_CONFIG['batch_size']
    seq_len = min(config.DATA_CONFIG['max_seq_len'], config.TRAINING_CONFIG['seq_len'])
    input_dim = config.MODEL_CONFIG['input_dim']
    noise_factor = config.DATA_CONFIG['noise_factor']

    # Datos de entrenamiento
    X_train = torch.randn(train_size, seq_len, input_dim)
    # Target: reconstrucción con ruido (autoencoder-like)
    X_train_target = X_train + noise_factor * torch.randn_like(X_train)

    # Datos de validación
    X_val = torch.randn(val_size, seq_len, input_dim)
    X_val_target = X_val + noise_factor * torch.randn_like(X_val)

    # Crear datasets
    train_dataset = TensorDataset(X_train, X_train_target)
    val_dataset = TensorDataset(X_val, X_val_target)

    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"   ✅ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   ✅ Val: {len(val_dataset)} samples, {len(val_loader)} batches")

def create_text_like_data(config):
    """Crea datos que simulan embeddings de texto real"""
    print("📚 Generating realistic text-like data...")

    train_size = config.DATA_CONFIG['train_samples']
    val_size = config.DATA_CONFIG['val_samples']
    batch_size = config.TRAINING_CONFIG['batch_size']
    seq_len = min(config.DATA_CONFIG['max_seq_len'], config.TRAINING_CONFIG['seq_len'])
    input_dim = config.MODEL_CONFIG['input_dim']
    noise_factor = config.DATA_CONFIG['noise_factor']

    # Datos de entrenamiento con patrones realistas
    X_train = generate_structured_text_data(train_size, seq_len, input_dim)

    # Target: reconstrucción con ruido más realista
    semantic_noise = generate_semantic_noise(X_train, noise_factor)
    X_train_target = X_train + semantic_noise

    # Datos de validación
    X_val = generate_structured_text_data(val_size, seq_len, input_dim)
    val_semantic_noise = generate_semantic_noise(X_val, noise_factor)
    X_val_target = X_val + val_semantic_noise

    # Crear datasets
    train_dataset = TensorDataset(X_train, X_train_target)
    val_dataset = TensorDataset(X_val, X_val_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"   ✅ Text-like: {len(train_dataset)} sequences with semantic structure")
    print(f"   ✅ Input dim: {input_dim} (emulating embeddings)")
    print(f"   ✅ Seq len: {seq_len} (sequence length)")

    return train_loader, val_loader


def generate_structured_text_data(num_samples, seq_len, input_dim):
    """Genera datos que simulan embeddings de texto con estructura semántica"""
    np.random.seed(42)  # Para reproducibilidad

    data = []

    for sample in range(num_samples):
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

            # Coherencia posicional y semántica
            position_factor = np.sin(pos / seq_len * 2 * np.pi) * 0.1
            embedding += position_factor

            if pos > 0:
                coherence = 0.3 * sequence[-1]
                embedding = 0.7 * embedding + coherence

            sequence.append(embedding)

        data.append(np.array(sequence))

    return torch.FloatTensor(np.array(data))


def generate_semantic_noise(data, noise_factor):
    """Genera ruido que respeta la estructura semántica"""
    batch_size, seq_len, input_dim = data.shape

    noise = torch.randn_like(data) * noise_factor
    importance_mask = torch.linspace(1.0, 0.3, input_dim)
    importance_mask = importance_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

def create_complex_text_data(config):
    """Crea datos más complejos que simulen texto real con mayor variabilidad"""
    print("🧠 Generating complex text-like data...")

    train_size = config.DATA_CONFIG['train_samples']
    val_size = config.DATA_CONFIG['val_samples']
    batch_size = config.TRAINING_CONFIG['batch_size']
    seq_len = min(config.DATA_CONFIG['max_seq_len'], config.TRAINING_CONFIG['seq_len'])
    input_dim = config.MODEL_CONFIG['input_dim']
    noise_factor = config.DATA_CONFIG['noise_factor']

    # Datos con mayor complejidad y variabilidad
    X_train = generate_complex_text_embeddings(train_size, seq_len, input_dim)
    X_train_target = X_train + generate_adaptive_noise(X_train, noise_factor)

    # Datos de validación
    X_val = generate_complex_text_embeddings(val_size, seq_len, input_dim)
    X_val_target = X_val + generate_adaptive_noise(X_val, noise_factor)

    train_dataset = TensorDataset(X_train, X_train_target)
    val_dataset = TensorDataset(X_val, X_val_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"   ✅ Complex text: {len(train_dataset)} sequences with high variability")
    print(f"   ✅ Input dim: {input_dim}, Seq len: {seq_len}")
    print(f"   ✅ Adaptive noise: {noise_factor}")

    return train_loader, val_loader


def generate_complex_text_embeddings(num_samples, seq_len, input_dim):
    """Genera embeddings más complejos que simulan texto real con mayor diversidad"""
    np.random.seed(42)

    data = []

    for sample in range(num_samples):
        sequence = []

        # Crear contexto global para cada secuencia
        global_theme = np.random.choice(['technical', 'narrative', 'dialogue', 'descriptive'], p=[0.3, 0.3, 0.2, 0.2])

        for pos in range(seq_len):
            # Más tipos de tokens con distribuciones más realistas
            token_types = ['noun', 'verb', 'adj', 'adv', 'prep', 'pron', 'conj', 'punct']
            probabilities = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]
            token_type = np.random.choice(token_types, p=probabilities)

            # Embeddings más diversos según el tipo
            if token_type == 'noun':
                embedding = np.random.normal(0.3, 0.4, input_dim)
            elif token_type == 'verb':
                embedding = np.random.normal(0.1, 0.6, input_dim)
            elif token_type == 'adj':
                embedding = np.random.normal(-0.2, 0.5, input_dim)
            elif token_type == 'adv':
                embedding = np.random.normal(0.0, 0.3, input_dim)
            elif token_type == 'prep':
                embedding = np.random.normal(-0.1, 0.2, input_dim)
            else:
                embedding = np.random.normal(0, 0.15, input_dim)

            # Añadir variaciones contextuales
            if global_theme == 'technical':
                embedding += np.random.normal(0.1, 0.1, input_dim)  # Más preciso
            elif global_theme == 'narrative':
                embedding += np.random.normal(0.2, 0.3, input_dim)  # Más expresivo
            elif global_theme == 'dialogue':
                embedding += np.random.normal(-0.1, 0.4, input_dim)  # Más coloquial
            else:  # descriptive
                embedding += np.random.normal(0.0, 0.5, input_dim)  # Más variado

            # Coherencia posicional más compleja
            position_encoding = np.sin(pos / seq_len * 2 * np.pi) * 0.1
            position_encoding += np.cos(pos / seq_len * 4 * np.pi) * 0.05
            embedding += position_encoding

            # Coherencia semántica con memoria a más largo plazo
            if pos > 0:
                # Influencia del token anterior (fuerte)
                embedding = 0.4 * embedding + 0.3 * sequence[-1]
                # Influencia de tokens anteriores (débil)
                if len(sequence) > 2:
                    embedding += 0.2 * sequence[-3]
                if len(sequence) > 5:
                    embedding += 0.1 * sequence[-6]

            # Añadir variabilidad intra-token (como sinónimos)
            synonym_variation = np.random.normal(0, 0.1, input_dim)
            embedding += synonym_variation

            sequence.append(embedding)

        data.append(np.array(sequence))

    return torch.FloatTensor(np.array(data))


def generate_adaptive_noise(data, noise_factor):
    """Genera ruido que se adapta a la complejidad local de los datos"""
    batch_size, seq_len, input_dim = data.shape

    # Calcular complejidad local (variabilidad en ventanas)
    complexity_map = torch.zeros_like(data)

    for b in range(batch_size):
        for s in range(seq_len):
            # Ventana de contexto
            start = max(0, s-2)
            end = min(seq_len, s+3)
            window = data[b, start:end]

            # Complejidad = varianza en la ventana
            if window.shape[0] > 1:
                complexity = window.var(dim=0)
                complexity_map[b, s] = complexity

    # Normalizar complejidad
    complexity_map = complexity_map / (complexity_map.max() + 1e-8)

    # Generar ruido base
    noise = torch.randn_like(data) * noise_factor

    # Modificar ruido según complejidad: más ruido donde hay menos complejidad
    adaptive_noise = noise * (1.0 + complexity_map * 0.5)

    return adaptive_noise


def create_variable_length_data(config):
    """Crea datos con longitudes variables para entrenamiento más robusto"""
    print("🔄 Generating variable length data...")

    train_size = config.DATA_CONFIG['train_samples']
    val_size = config.DATA_CONFIG['val_samples']
    batch_size = config.TRAINING_CONFIG['batch_size']
    min_seq_len = config.DATA_CONFIG.get('min_seq_len', 8)
    max_seq_len = config.DATA_CONFIG['max_seq_len']
    input_dim = config.MODEL_CONFIG['input_dim']
    noise_factor = config.DATA_CONFIG['noise_factor']

    # Generar secuencias con longitudes variables
    train_sequences = []
    val_sequences = []

    for _ in range(train_size):
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        X = generate_complex_text_embeddings(1, seq_len, input_dim).squeeze(0)
        noise = generate_adaptive_noise(X.unsqueeze(0), noise_factor).squeeze(0)
        train_sequences.append((X, X + noise))

    for _ in range(val_size):
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        X = generate_complex_text_embeddings(1, seq_len, input_dim).squeeze(0)
        noise = generate_adaptive_noise(X.unsqueeze(0), noise_factor).squeeze(0)
        val_sequences.append((X, X + noise))

    # Usar padding para crear batches uniformes
    train_dataset = VariableLengthDataset(train_sequences, input_dim)
    val_dataset = VariableLengthDataset(val_sequences, input_dim)

    # Collate function para manejar longitudes variables
    def collate_fn(batch):
        # batch = [(X1, y1), (X2, y2), ...]
        max_len = max(len(x) for x, y in batch)

        # Padding
        padded_X = []
        padded_y = []
        lengths = []

        for x, y in batch:
            length = len(x)
            lengths.append(length)

            # Padding con ceros
            pad_size = max_len - length
            if pad_size > 0:
                pad = torch.zeros(pad_size, input_dim)
                x_padded = torch.cat([x, pad], dim=0)
                y_padded = torch.cat([y, pad], dim=0)
            else:
                x_padded = x
                y_padded = y

            padded_X.append(x_padded)
            padded_y.append(y_padded)

        return torch.stack(padded_X), torch.stack(padded_y), torch.tensor(lengths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    print(f"   ✅ Variable length: {len(train_dataset)} sequences")
    print(f"   ✅ Length range: {min_seq_len}-{max_seq_len}")
    print(f"   ✅ With dynamic padding")

    return train_loader, val_loader


class VariableLengthDataset(torch.utils.data.Dataset):
    """Dataset que maneja secuencias de diferentes longitudes"""

    def __init__(self, sequences, input_dim):
        self.sequences = sequences
        self.input_dim = input_dim

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def train_with_variable_lengths(config):
    """Entrenamiento que maneja longitudes variables"""
    print("🌟 Pure DFN Variable Length Training")
    print("=" * 50)
    print(f"📊 Model parameters: {sum(p.numel() for p in DFNPureModel(**config.MODEL_CONFIG).parameters()):,}")
    print(f"🎯 Epochs: {config.TRAINING_CONFIG['num_epochs']}")
    print(f"📦 Batch size: {config.TRAINING_CONFIG['batch_size']}")
    print(f"💾 Output dir: {config.TRAINING_CONFIG['output_dir']}")
    print()

    # Crear modelo
    model, device = create_model(config)

    # Crear datos con longitudes variables
    train_loader, val_loader = create_variable_length_data(config)

    # Configurar entrenamiento
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAINING_CONFIG['learning_rate'],
        weight_decay=config.TRAINING_CONFIG['weight_decay']
    )

    # Scheduler
    total_steps = len(train_loader) * config.TRAINING_CONFIG['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.TRAINING_CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1
    )

    # Entrenamiento con manejo de máscaras
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': [], 'memory_usage': []}

    for epoch in range(config.TRAINING_CONFIG['num_epochs']):
        start_time = time.time()

        # Entrenar época
        train_loss = train_epoch_variable_length(model, train_loader, optimizer, criterion, scheduler, config, epoch, device)

        # Validar
        val_loss = validate_variable_length(model, val_loader, criterion, device)

        # Actualizar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        # GPU memory
        if torch.cuda.is_available():
            history['memory_usage'].append({
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
            })

        # Checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config)

        if (epoch + 1) % config.TRAINING_CONFIG['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config)

        # Resultados
        epoch_time = time.time() - start_time
        print(f"\n📈 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        print_gpu_memory_info("   🎮 Final GPU Memory:")

    # Guardar historial
    history_file = f"{config.TRAINING_CONFIG['output_dir']}/training_history_variable.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📋 History saved: {history_file}")

    # Resumen
    print("\n🏆 VARIABLE TRAINING COMPLETED!")
    print("=" * 50)
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Epochs: {config.TRAINING_CONFIG['num_epochs']}")
    print(f"   Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Checkpoints: {config.TRAINING_CONFIG['output_dir']}/")

    print_memory_summary(model, "📊 FINAL TRAINING SUMMARY")

    return model, history


def train_epoch_variable_length(model, train_loader, optimizer, criterion, scheduler, config, epoch, device):
    """Entrena una época con longitudes variables"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    print(f"\n🚀 Epoch {epoch+1}/{config.TRAINING_CONFIG['num_epochs']}")
    print_gpu_memory_info("   🎮 Memory before epoch:")

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, (x, y, lengths) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Crear máscara para padding (1 para tokens reales, 0 para padding)
        max_len = x.shape[1]
        mask = torch.zeros_like(x)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1

        # Forward pass
        optimizer.zero_grad()
        output_field = model(x)

        # Calcular pérdida solo en tokens reales (sin padding)
        loss = masked_loss(output_field.mean(dim=1), y.mean(dim=1), mask.mean(dim=1))

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAINING_CONFIG['gradient_clip'])
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}",
            'gpu_mb': f"{torch.cuda.memory_allocated()/1024/1024:.1f}" if torch.cuda.is_available() else "CPU"
        })

        if batch_idx % 50 == 0:
            print_gpu_memory_info(f"   📊 Batch {batch_idx}:")

    avg_loss = total_loss / num_batches
    print_gpu_memory_info("   🎮 Memory after epoch:")

    return avg_loss


def masked_loss(output, target, mask):
    """Calcula pérdida solo en posiciones no masked"""
    loss = torch.nn.functional.mse_loss(output * mask, target * mask, reduction='none')
    return loss.sum() / mask.sum()


def validate_variable_length(model, val_loader, criterion, device):
    """Valida con longitudes variables"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y, lengths in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)

            # Crear máscara
            max_len = x.shape[1]
            mask = torch.zeros_like(x)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1

            # Forward
            output_field = model(x)

            # Loss con máscara
            loss = masked_loss(output_field.mean(dim=1), y.mean(dim=1), mask.mean(dim=1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def create_model(config):
    """Crea el modelo DFN Pura"""
    print("🏗️  Creating Pure DFN model...")

    model = DFNPureModel(**config.MODEL_CONFIG)

    # Mover a GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   🔥 Device: {device}")

    # Mostrar información de memoria
    memory_info = get_model_memory_footprint(model)
    print(f"   📊 Memory: {memory_info['total_mb']:.1f}MB")

    return model, device


def train_epoch(model, train_loader, optimizer, criterion, scheduler, config, epoch):
    """Entrena una época completa"""
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    num_batches = 0

    print(f"\n🚀 Epoch {epoch+1}/{config.TRAINING_CONFIG['num_epochs']}")
    print_gpu_memory_info("   🎮 Memory before epoch:")

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, (x, y) in enumerate(pbar):
        # Mover datos a GPU
        x, y = x.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        output_field = model(x)

        # Calcular pérdida de reconstrucción
        # Promedio sobre el campo de salida
        loss = criterion(output_field.mean(dim=1), y.mean(dim=1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAINING_CONFIG['gradient_clip'])

        optimizer.step()
        scheduler.step()

        # Actualizar métricas
        total_loss += loss.item()
        num_batches += 1

        # Actualizar progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}",
            'gpu_mb': f"{torch.cuda.memory_allocated()/1024/1024:.1f}" if torch.cuda.is_available() else "CPU"
        })

        # Mostrar VRAM cada 50 batches
        if batch_idx % 50 == 0:
            print_gpu_memory_info(f"   📊 Batch {batch_idx}:")

    avg_loss = total_loss / num_batches
    print_gpu_memory_info("   🎮 Memory after epoch:")

    return avg_loss


def validate_model(model, val_loader, criterion, config):
    """Valida el modelo"""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            output_field = model(x)

            # Calcular pérdida
            loss = criterion(output_field.mean(dim=1), y.mean(dim=1))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config):
    """Guarda checkpoint del modelo"""
    os.makedirs(config.TRAINING_CONFIG['output_dir'], exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config.MODEL_CONFIG
    }

    filename = f"{config.TRAINING_CONFIG['output_dir']}/dfn_pure_epoch_{epoch+1:03d}_loss_{loss:.4f}.pt"
    torch.save(checkpoint, filename)
    print(f"💾 Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Carga checkpoint del modelo"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"📁 Checkpoint loaded: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Loss: {checkpoint['loss']:.4f}")
    return checkpoint['epoch'], checkpoint['loss']


# =============================================================================
# 🚀  FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# =============================================================================

def train_dfn(config):
    """Función principal de entrenamiento"""
    print("🌟 Pure DFN Training Script")
    print("=" * 50)
    print(f"📊 Model parameters: {sum(p.numel() for p in DFNPureModel(**config.MODEL_CONFIG).parameters()):,}")
    print(f"🎯 Epochs: {config.TRAINING_CONFIG['num_epochs']}")
    print(f"📦 Batch size: {config.TRAINING_CONFIG['batch_size']}")
    print(f"💾 Output dir: {config.TRAINING_CONFIG['output_dir']}")
    print()

    # Crear modelo
    model, device = create_model(config)

    # Crear datos según el tipo configurado
    data_type = config.DATA_CONFIG['data_type']
    if data_type == 'variable_length':
        model, history = train_with_variable_lengths(config)
    elif data_type == 'complex_text':
        train_loader, val_loader = create_complex_text_data(config)
        model, history = train_dfn_fixed_length(config, train_loader, val_loader)
    elif data_type == 'text_like':
        train_loader, val_loader = create_text_like_data(config)
        model, history = train_dfn_fixed_length(config, train_loader, val_loader)
    elif data_type == 'random':
        train_loader, val_loader = create_training_data(config)
        model, history = train_dfn_fixed_length(config, train_loader, val_loader)
    else:
        print(f"⚠️  Data type '{data_type}' not recognized, using text_like")
        train_loader, val_loader = create_text_like_data(config)
        model, history = train_dfn_fixed_length(config, train_loader, val_loader)

    return model, history


def train_dfn_fixed_length(config, train_loader, val_loader):
    """Versión de entrenamiento con longitud fija"""
    print("🌟 Pure DFN Fixed Length Training")
    print("=" * 50)
    print(f"📊 Model parameters: {sum(p.numel() for p in DFNPureModel(**config.MODEL_CONFIG).parameters()):,}")
    print(f"🎯 Epochs: {config.TRAINING_CONFIG['num_epochs']}")
    print(f"📦 Batch size: {config.TRAINING_CONFIG['batch_size']}")
    print(f"💾 Output dir: {config.TRAINING_CONFIG['output_dir']}")
    print()

    # Crear modelo
    model, device = create_model(config)

    # Configurar entrenamiento
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAINING_CONFIG['learning_rate'],
        weight_decay=config.TRAINING_CONFIG['weight_decay']
    )

    # Scheduler con warmup y cosine decay
    total_steps = len(train_loader) * config.TRAINING_CONFIG['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.TRAINING_CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1  # 10% warmup
    )

    # Historial de entrenamiento
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'memory_usage': []
    }

    # Entrenamiento principal
    best_val_loss = float('inf')

    for epoch in range(config.TRAINING_CONFIG['num_epochs']):
        start_time = time.time()

        # Entrenar época
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, config, epoch)

        # Validar
        val_loss = validate_model(model, val_loader, criterion, config)

        # Actualizar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        # GPU memory usage
        if torch.cuda.is_available():
            gpu_info = torch.cuda.memory_stats()
            history['memory_usage'].append({
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
            })

        # Guardar checkpoint si es mejor
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config)

        # Guardar checkpoint regular
        if (epoch + 1) % config.TRAINING_CONFIG['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config)

        # Mostrar resultados de época
        epoch_time = time.time() - start_time
        print(f"\n📈 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
        print_gpu_memory_info("   🎮 Final GPU Memory:")

    # Guardar historial
    history_file = f"{config.TRAINING_CONFIG['output_dir']}/training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📋 History saved: {history_file}")

    # Resumen final
    print("\n🏆 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Epochs: {config.TRAINING_CONFIG['num_epochs']}")
    print(f"   Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Checkpoints: {config.TRAINING_CONFIG['output_dir']}/")

    # Mostrar memoria final
    print_memory_summary(model, "📊 FINAL TRAINING SUMMARY")

    return model, history


# =============================================================================
# 🎯  EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    # Configuración de entrenamiento
    config = TrainingConfig()

    # Ejecutar entrenamiento
    try:
        model, history = train_dfn(config)
        print("\n✅ Training successful!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
