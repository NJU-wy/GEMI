import unittest
import numpy as np
import torch

from config import LABEL_NORMAL, LABEL_ISC, LABEL_LOW_CAP, ISC_CELLS, LOW_CAP_CELLS
from main import split_cells, compute_pack_mean, normalize_macro, build_slices


class TestPipelineUnits(unittest.TestCase):
    def test_split_cells_disjoint(self):
        labels = np.array([LABEL_NORMAL, LABEL_NORMAL, LABEL_ISC, LABEL_LOW_CAP])
        train_idx, test_idx = split_cells(labels, seed=123, train_ratio=0.5)
        self.assertEqual(len(set(train_idx).intersection(set(test_idx))), 0)
        self.assertEqual(set(train_idx).union(set(test_idx)), set(range(len(labels))))

    def test_compute_pack_mean_train_only(self):
        V = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0]
        ], dtype=np.float32)
        labels = np.array([LABEL_NORMAL, LABEL_NORMAL])
        train_idx = np.array([0])
        pack_mean = compute_pack_mean(V, labels, train_idx)
        expected = V[:, [0]]
        self.assertTrue(np.allclose(pack_mean, expected))

    def test_normalize_macro_train_stats(self):
        macro = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]], dtype=np.float32)
        train_mask = np.array([True, True, False])
        normalized, mean, std = normalize_macro(macro, train_mask)
        expected_mean = np.array([2.0, 3.0], dtype=np.float32)
        self.assertTrue(np.allclose(mean, expected_mean))
        self.assertTrue(np.allclose((macro - mean) / std, normalized))

    def test_build_slices_labels(self):
        cell_names = [ISC_CELLS[0], LOW_CAP_CELLS[0]]
        V = np.array([
            [1.0, 2.0],
            [1.2, 2.2],
            [1.1, 2.1],
            [1.3, 2.3],
            [1.2, 2.2],
            [1.4, 2.4]
        ], dtype=np.float32)
        I = np.ones(6, dtype=np.float32)
        S = np.zeros(6, dtype=np.float32)
        T = np.zeros(6, dtype=np.float32) + 25.0
        pack_mean = np.mean(V[:, [0]], axis=1, keepdims=True)
        raw, for_ae, phy, macro, labels, cell_idx = build_slices(
            V, I, S, T, cell_names, pack_mean, fault_threshold=0.05, dynamic_i_threshold=0.5, win=5
        )
        self.assertIn(LABEL_ISC, labels)
        self.assertIn(LABEL_LOW_CAP, labels)


class TestPipelineIntegration(unittest.TestCase):
    def test_end_to_end_shapes(self):
        cell_names = ["C0-1", "C0-2"]
        V = np.random.randn(12, 2).astype(np.float32)
        I = np.random.randn(12).astype(np.float32)
        S = np.random.rand(12).astype(np.float32)
        T = np.random.rand(12).astype(np.float32) * 10 + 20
        labels = np.array([LABEL_NORMAL, LABEL_NORMAL])
        train_idx, _ = split_cells(labels, seed=1, train_ratio=0.5)
        pack_mean = compute_pack_mean(V, labels, train_idx)
        raw, for_ae, phy, macro, slice_labels, cell_idx = build_slices(
            V, I, S, T, cell_names, pack_mean, fault_threshold=10.0, dynamic_i_threshold=10.0, win=6
        )
        train_mask = np.isin(cell_idx, train_idx)
        macro_norm, _, _ = normalize_macro(macro, train_mask)
        self.assertEqual(raw.shape[0], for_ae.shape[0])
        self.assertEqual(raw.shape[0], phy.shape[0])
        self.assertEqual(raw.shape[0], macro_norm.shape[0])
        self.assertEqual(raw.shape[1], 6)
        self.assertEqual(macro_norm.shape[1], 7)


if __name__ == "__main__":
    unittest.main()
