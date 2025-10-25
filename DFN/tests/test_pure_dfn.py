"""
Tests for Pure DFN Model

Tests específicos para DFNPureModel que no usa vocab_size ni max_seq_len.
"""

import torch
import torch.nn as nn
import unittest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfn.models import DFNPureModel


class TestPureDFN(unittest.TestCase):
    """Tests for pure DFN implementation"""

    def setUp(self):
        """Setup test parameters"""
        self.batch_size = 2
        self.seq_len = 20
        self.input_dim = 64

    def test_pure_dfn_no_vocab_size(self):
        """Test that DFNPureModel doesn't require vocab_size"""
        # Crear modelo sin vocab_size ni max_seq_len
        model = DFNPureModel(
            input_dim=self.input_dim,
            entity_dim=128,
            field_dim=64,
            num_entities=16,
            num_layers=2,
            coord_dim=2
        )

        # Crear entrada de cualquier longitud y dimensionalidad
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Forward pass debería funcionar sin problemas
        output = model(x)

        # Verificar shapes
        self.assertEqual(len(output.shape), 3)
        self.assertEqual(output.shape[0], self.batch_size)  # batch dimension
        self.assertEqual(output.shape[2], self.input_dim)   # output dimension

        print(f"✓ Pure DFN works without vocab_size: {x.shape} -> {output.shape}")

    def test_variable_sequence_length(self):
        """Test that model handles variable sequence lengths"""
        model = DFNPureModel(
            input_dim=self.input_dim,
            entity_dim=64,
            num_entities=8,
            num_layers=1
        )

        # Secuencias de diferentes longitudes
        lengths = [10, 50, 100, 200]

        for length in lengths:
            x = torch.randn(self.batch_size, length, self.input_dim)
            output = model(x)

            # La salida debería tener siempre la misma estructura
            self.assertEqual(output.shape[0], self.batch_size)
            self.assertEqual(output.shape[2], self.input_dim)

            print(f"✓ Handles sequence length {length}: {x.shape} -> {output.shape}")

    def test_entity_extraction(self):
        """Test continuous entity extraction"""
        model = DFNPureModel(
            input_dim=self.input_dim,
            entity_dim=64,
            num_entities=8,
            num_layers=1
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Obtener entidades
        positions, states, weights = model.encode(x)

        # Verificar shapes de entidades
        self.assertEqual(positions.shape[0], self.batch_size)
        self.assertEqual(positions.shape[1], 8)  # num_entities
        self.assertEqual(positions.shape[2], 2)  # coord_dim

        self.assertEqual(states.shape[0], self.batch_size)
        self.assertEqual(states.shape[1], 8)    # num_entities
        self.assertEqual(states.shape[2], 64)   # entity_dim

        self.assertEqual(weights.shape[0], self.batch_size)
        self.assertEqual(weights.shape[1], 8)   # num_entities
        self.assertEqual(weights.shape[2], 1)   # scalar weight

        # Verificar que los pesos suman a 1 (normalizados)
        weight_sum = weights.squeeze(-1).sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sum, torch.ones(self.batch_size), atol=1e-6))

        print(f"✓ Entity extraction: {states.shape} entities with weights summing to 1")

    def test_density_field_adaptation(self):
        """Test that density field adapts to information density"""
        model = DFNPureModel(
            input_dim=self.input_dim,
            entity_dim=64,
            num_entities=8,
            num_layers=1
        )

        # Crear datos con diferente densidad de información
        # Patrón 1: información uniforme
        x_uniform = torch.randn(self.batch_size, 30, self.input_dim) * 0.5

        # Patrón 2: información concentrada en la segunda mitad
        x_concentrated = torch.randn(self.batch_size, 30, self.input_dim)
        x_concentrated[:, 15:, :] *= 3.0  # Más información

        # Extraer entidades de ambos patrones
        pos1, states1, weights1 = model.encode(x_uniform)
        pos2, states2, weights2 = model.encode(x_concentrated)

        # Los pesos deberían reflejar la distribución de información
        avg_weights1 = weights1.squeeze(-1).mean(dim=0)  # Promedio sobre batches
        avg_weights2 = weights2.squeeze(-1).mean(dim=0)

        # El patrón concentrado debería tener mayor varianza en pesos
        variance1 = torch.var(avg_weights1)
        variance2 = torch.var(avg_weights2)

        self.assertGreater(variance2, variance1 * 1.5)  # Mayor varianza en patrón concentrado

        print(f"✓ Density adaptation: uniform variance={variance1:.4f}, concentrated variance={variance2:.4f}")

    def test_memory_persistence(self):
        """Test that memory persists across forward passes"""
        model = DFNPureModel(
            input_dim=self.input_dim,
            entity_dim=64,
            num_entities=8,
            num_layers=2,
            memory_size=16
        )

        # Primera secuencia
        x1 = torch.randn(self.batch_size, 20, self.input_dim)
        output1 = model(x1)

        # Verificar que memoria se actualizó
        memory_before = []
        for layer in model.dfn_layers:
            memory_before.append(layer.memory.memory.clone())

        # Segunda secuencia
        x2 = torch.randn(self.batch_size, 25, self.input_dim)
        output2 = model(x2)

        # Verificar que memoria cambió
        memory_changed = False
        for i, layer in enumerate(model.dfn_layers):
            if not torch.allclose(memory_before[i], layer.memory.memory):
                memory_changed = True
                break

        self.assertTrue(memory_changed, "Memory should update across forward passes")

        # Reset memory
        model.reset_memory()
        for layer in model.dfn_layers:
            self.assertTrue(torch.allclose(layer.memory.memory, torch.zeros_like(layer.memory.memory)))

        print("✓ Memory persistence and reset working correctly")

    def test_continuous_output_field(self):
        """Test continuous output field generation"""
        model = DFNPureModel(
            input_dim=self.input_dim,
            entity_dim=64,
            num_entities=8,
            num_layers=1,
            field_resolution=32
        )

        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        # Generar campo de salida
        output_field = model(x)

        expected_shape = (self.batch_size, 32, self.input_dim)  # field_resolution=32
        self.assertEqual(output_field.shape, expected_shape)

        # Verificar que la salida es continua (no discreta)
        # La salida debería variar suavemente, no tener valores discretos
        self.assertGreater(output_field.std(), 0.1)  # Debería tener varianza significativa

        print(f"✓ Continuous output field: {output_field.shape}, std={output_field.std().item():.4f}")

    def test_coordinate_generation(self):
        """Test that coordinates are generated from content"""
        model = DFNPureModel(
            input_dim=self.input_dim,
            entity_dim=64,
            num_entities=8,
            num_layers=1
        )

        # Crear dos entradas diferentes
        x1 = torch.randn(self.batch_size, 20, self.input_dim)
        x2 = torch.randn(self.batch_size, 20, self.input_dim)

        # Generar coordenadas
        coords1, densities1 = model.density_field(x1)
        coords2, densities2 = model.density_field(x2)

        # Las coordenadas deberían ser diferentes para entradas diferentes
        coord_diff = torch.norm(coords1 - coords2).item()
        self.assertGreater(coord_diff, 0.1)  # Deberían ser significativamente diferentes

        print(f"✓ Content-based coordinate generation: difference={coord_diff:.4f}")


def run_pure_dfn_tests():
    """Run pure DFN tests"""
    print("Testing Pure DFN Implementation...")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPureDFN))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All pure DFN tests passed!")
        print("\nPure DFN characteristics verified:")
        print("- ✅ No vocab_size dependency")
        print("- ✅ Variable sequence length support")
        print("- ✅ Continuous entity extraction")
        print("- ✅ Density-based adaptation")
        print("- ✅ Persistent memory")
        print("- ✅ Continuous output fields")
        print("- ✅ Content-based coordinates")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_pure_dfn_tests()
    sys.exit(0 if success else 1)
