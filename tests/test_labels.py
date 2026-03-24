import unittest
import numpy as np

from config import (
    LABEL_NORMAL,
    LABEL_ISC,
    LABEL_LOW_CAP,
    ISC_CELLS,
    LOW_CAP_CELLS,
    get_cell_label,
    build_global_labels,
    validate_label_config,
)


class TestLabelConfig(unittest.TestCase):
    def test_validate_label_config(self):
        self.assertTrue(validate_label_config())

    def test_label_lists_exact(self):
        expected_isc = {"C9-17", "C7-16"}
        expected_low_cap = {
            "C9-15", "C9-7", "C8-11", "C8-5", "C7-11", "C7-5",
            "C6-11", "C6-5", "C5-11", "C5-5", "C4-7", "C3-15",
            "C3-7", "C2-7",
        }
        self.assertEqual(set(ISC_CELLS), expected_isc)
        self.assertEqual(set(LOW_CAP_CELLS), expected_low_cap)

    def test_get_cell_label_faults(self):
        for name in ISC_CELLS:
            self.assertEqual(get_cell_label(name), LABEL_ISC)
        for name in LOW_CAP_CELLS:
            self.assertEqual(get_cell_label(name), LABEL_LOW_CAP)

    def test_get_cell_label_normal(self):
        normal_names = ["C0-0", "C2-8", "C9-18"]
        for name in normal_names:
            self.assertEqual(get_cell_label(name), LABEL_NORMAL)

    def test_build_global_labels(self):
        cell_names = ISC_CELLS + LOW_CAP_CELLS + ["C0-0"]
        labels = build_global_labels(cell_names)
        expected = (
            [LABEL_ISC] * len(ISC_CELLS)
            + [LABEL_LOW_CAP] * len(LOW_CAP_CELLS)
            + [LABEL_NORMAL]
        )
        self.assertTrue(np.array_equal(labels, np.array(expected, dtype=int)))


if __name__ == "__main__":
    unittest.main()
