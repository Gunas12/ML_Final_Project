"""Unit tests for the from-scratch Pegasos SVM public prediction API."""

import unittest

import numpy as np

from src.svm import PegasosSVM


class TestPegasosSVMLabels(unittest.TestCase):
    def test_binary_labels_are_encoded_and_mapping_is_stored(self):
        model = PegasosSVM()

        encoded = model._to_pm1(np.array([1, 0, 1, 0]))

        np.testing.assert_array_equal(encoded, [1.0, -1.0, 1.0, -1.0])
        np.testing.assert_array_equal(model.classes_, [0, 1])
        self.assertEqual(model.class_to_pm1_, {0: -1.0, 1: 1.0})

    def test_arbitrary_binary_labels_are_supported(self):
        model = PegasosSVM()

        encoded = model._to_pm1(["standard", "premium", "premium"])

        np.testing.assert_array_equal(encoded, [1.0, -1.0, -1.0])
        np.testing.assert_array_equal(model.classes_, ["premium", "standard"])

    def test_non_binary_labels_are_rejected(self):
        for labels in ([1], [0, 1, 2]):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(ValueError, "exactly two classes"):
                    PegasosSVM()._to_pm1(labels)


class TestPegasosSVMPrediction(unittest.TestCase):
    def setUp(self):
        self.model = PegasosSVM()
        self.model._to_pm1(np.array(["standard", "premium"]))
        self.model.w = np.array([2.0, -1.0])
        self.model.b = 0.5

    def test_decision_function_returns_signed_margins(self):
        X = np.array([[1.0, 1.0], [0.0, 2.0], [0.0, 0.5]])

        margins = self.model.decision_function(X)

        np.testing.assert_allclose(margins, [1.5, -1.5, 0.0])

    def test_predict_restores_original_labels(self):
        X = np.array([[1.0, 1.0], [0.0, 2.0], [0.0, 0.5]])

        predictions = self.model.predict(X)

        np.testing.assert_array_equal(
            predictions, ["standard", "premium", "standard"]
        )

    def test_prediction_requires_a_fitted_model(self):
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            PegasosSVM().predict([[1.0, 2.0]])

    def test_feature_count_must_match_weights(self):
        with self.assertRaisesRegex(ValueError, "number of features"):
            self.model.decision_function([[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
