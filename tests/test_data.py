import unittest

from prediccion_fuga.data import FEATURES, generar_dataset


class DataContractTest(unittest.TestCase):
    def test_dataset_is_reproducible_and_has_expected_schema(self):
        first = generar_dataset(20, seed=123)
        second = generar_dataset(20, seed=123)
        self.assertTrue(first.equals(second))
        self.assertEqual(first.columns.tolist(), FEATURES + ["se_fue"])
        self.assertTrue(first["se_fue"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
