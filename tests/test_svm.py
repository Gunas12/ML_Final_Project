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


class TestPegasosSVMFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        negative = rng.normal(loc=-2.0, scale=0.35, size=(50, 2))
        positive = rng.normal(loc=2.0, scale=0.35, size=(50, 2))
        cls.X = np.vstack([negative, positive])
        cls.y = np.array(["standard"] * 50 + ["premium"] * 50)

    def make_model(self):
        return PegasosSVM(lambda_=0.1, n_iters=4_000, random_state=7)

    def test_fit_learns_separable_toy_data_and_reduces_loss(self):
        model = self.make_model().fit(self.X, self.y)

        accuracy = np.mean(model.predict(self.X) == self.y)
        self.assertGreaterEqual(accuracy, 0.95)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])
        self.assertEqual(model.loss_steps_[0], 0)
        self.assertEqual(model.loss_steps_[-1], model.n_iters)
        self.assertEqual(len(model.loss_history_), len(model.loss_steps_))

    def test_same_seed_produces_same_parameters(self):
        first = self.make_model().fit(self.X, self.y)
        second = self.make_model().fit(self.X, self.y)

        np.testing.assert_array_equal(first.w, second.w)
        self.assertEqual(first.b, second.b)

    def test_fit_returns_self(self):
        model = self.make_model()

        result = model.fit(self.X, self.y)

        self.assertIs(result, model)

    def test_invalid_X_y_dimensions_are_rejected(self):
        invalid_inputs = [
            (np.array([1.0, 2.0]), np.array([0, 1]), "two-dimensional"),
            (np.ones((2, 2)), np.array([[0], [1]]), "one-dimensional"),
            (np.ones((3, 2)), np.array([0, 1]), "same number"),
        ]

        for X, y, message in invalid_inputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_model().fit(X, y)

    def test_empty_and_non_finite_data_are_rejected(self):
        invalid_inputs = [
            (np.empty((0, 2)), np.array([]), "must not be empty"),
            (np.array([[0.0], [np.nan]]), np.array([0, 1]), "NaN"),
            (np.array([[0.0], [1.0]]), np.array([0.0, np.inf]), "NaN"),
        ]

        for X, y, message in invalid_inputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_model().fit(X, y)

    def test_invalid_hyperparameters_are_rejected(self):
        invalid_models = [
            PegasosSVM(lambda_=0.0, n_iters=10),
            PegasosSVM(lambda_=np.inf, n_iters=10),
            PegasosSVM(lambda_=0.1, n_iters=0),
            PegasosSVM(lambda_=0.1, n_iters=1.5),
        ]

        for model in invalid_models:
            with self.subTest(lambda_=model.lambda_, n_iters=model.n_iters):
                with self.assertRaises(ValueError):
                    model.fit(self.X, self.y)


if __name__ == "__main__":
    unittest.main()
