"""Unit tests for the from-scratch Pegasos SVM public prediction API."""

import unittest

import numpy as np

from src.svm import PegasosSVM, RandomFourierFeatures


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


class TestRandomFourierFeatures(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array(
            [[-1.0, 0.5], [0.0, 1.0], [1.5, -0.5], [2.0, 1.0]]
        )

    def test_output_shape_and_finite_values(self):
        transformed = RandomFourierFeatures(n_components=32).fit_transform(
            self.X_train
        )

        self.assertEqual(transformed.shape, (4, 32))
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_same_seed_reproduces_mapping_and_transform(self):
        first = RandomFourierFeatures(n_components=16, random_state=7).fit(
            self.X_train
        )
        second = RandomFourierFeatures(n_components=16, random_state=7).fit(
            self.X_train
        )

        np.testing.assert_array_equal(first.W_, second.W_)
        np.testing.assert_array_equal(first.offset_, second.offset_)
        np.testing.assert_array_equal(
            first.transform(self.X_train), second.transform(self.X_train)
        )

    def test_different_seeds_produce_different_transforms(self):
        first = RandomFourierFeatures(n_components=16, random_state=1)
        second = RandomFourierFeatures(n_components=16, random_state=2)

        first_result = first.fit_transform(self.X_train)
        second_result = second.fit_transform(self.X_train)

        self.assertFalse(np.array_equal(first_result, second_result))

    def test_transform_before_fit_raises(self):
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            RandomFourierFeatures().transform(self.X_train)

    def test_fit_returns_self(self):
        mapping = RandomFourierFeatures()

        self.assertIs(mapping.fit(self.X_train), mapping)

    def test_fit_transform_matches_fit_then_transform(self):
        combined = RandomFourierFeatures(n_components=24, random_state=11)
        separate = RandomFourierFeatures(n_components=24, random_state=11)

        combined_result = combined.fit_transform(self.X_train)
        separate_result = separate.fit(self.X_train).transform(self.X_train)

        np.testing.assert_array_equal(combined_result, separate_result)

    def test_validation_transform_reuses_training_mapping(self):
        mapping = RandomFourierFeatures(
            n_components=20, gamma=0.75, random_state=5
        ).fit(self.X_train)
        W_before = mapping.W_.copy()
        offset_before = mapping.offset_.copy()
        X_validation = np.array([[0.25, -0.75], [1.25, 0.5]])

        transformed = mapping.transform(X_validation)
        expected = np.sqrt(2.0 / 20) * np.cos(
            X_validation @ W_before + offset_before
        )

        np.testing.assert_allclose(transformed, expected)
        np.testing.assert_array_equal(mapping.W_, W_before)
        np.testing.assert_array_equal(mapping.offset_, offset_before)

    def test_invalid_hyperparameters_raise(self):
        invalid_mappings = [
            RandomFourierFeatures(n_components=0),
            RandomFourierFeatures(n_components=-1),
            RandomFourierFeatures(n_components=1.5),
            RandomFourierFeatures(n_components=True),
            RandomFourierFeatures(gamma=0.0),
            RandomFourierFeatures(gamma=-1.0),
            RandomFourierFeatures(gamma=np.inf),
            RandomFourierFeatures(gamma=True),
        ]

        for mapping in invalid_mappings:
            with self.subTest(
                n_components=mapping.n_components, gamma=mapping.gamma
            ):
                with self.assertRaises(ValueError):
                    mapping.fit(self.X_train)

    def test_invalid_fit_and_transform_inputs_raise(self):
        invalid_fit_inputs = [
            np.empty((0, 2)),
            np.empty((2, 0)),
            np.array([1.0, 2.0]),
            np.array([[0.0, np.nan]]),
            np.array([[0.0, np.inf]]),
        ]
        for X in invalid_fit_inputs:
            with self.subTest(stage="fit", shape=X.shape):
                with self.assertRaises(ValueError):
                    RandomFourierFeatures().fit(X)

        mapping = RandomFourierFeatures().fit(self.X_train)
        invalid_transform_inputs = [
            np.empty((0, 2)),
            np.array([1.0, 2.0]),
            np.array([[0.0, np.nan]]),
            np.array([[0.0, np.inf]]),
            np.ones((2, 3)),
        ]
        for X in invalid_transform_inputs:
            with self.subTest(stage="transform", shape=X.shape):
                with self.assertRaises(ValueError):
                    mapping.transform(X)

    def test_rff_with_pegasos_learns_xor(self):
        rng = np.random.default_rng(2026)
        centers = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
        labels = np.array([0, 0, 1, 1])
        X_parts = [rng.normal(center, 0.18, size=(80, 2)) for center in centers]
        X = np.vstack(X_parts)
        y = np.repeat(labels, 80)
        order = rng.permutation(len(y))
        X, y = X[order], y[order]
        X_train, X_test = X[:240], X[240:]
        y_train, y_test = y[:240], y[240:]

        mapping = RandomFourierFeatures(
            n_components=256, gamma=1.0, random_state=17
        )
        X_train_rff = mapping.fit_transform(X_train)
        X_test_rff = mapping.transform(X_test)
        model = PegasosSVM(lambda_=0.1, n_iters=12_000, random_state=29)
        model.fit(X_train_rff, y_train)

        accuracy = np.mean(model.predict(X_test_rff) == y_test)

        self.assertGreaterEqual(accuracy, 0.95)


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


class TestPegasosSVMDiagnostics(unittest.TestCase):
    def test_weight_norm_and_margin_width_for_known_weights(self):
        model = PegasosSVM()
        model.w = np.array([3.0, 4.0])

        self.assertEqual(model.weight_norm(), 5.0)
        self.assertEqual(model.margin_width(), 0.4)

    def test_zero_weights_have_infinite_margin_width(self):
        model = PegasosSVM()
        model.w = np.zeros(3)

        self.assertTrue(np.isinf(model.margin_width()))

    def test_support_mask_matches_hand_calculated_margins(self):
        model = PegasosSVM()
        model._to_pm1([0, 1])
        model.w = np.array([1.0, 0.0])
        model.b = 0.0
        X = np.array([[2.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [-0.5, 0.0]])
        y = np.array([1, 1, 0, 0])

        mask = model.support_mask(X, y)

        self.assertEqual(mask.dtype, np.bool_)
        np.testing.assert_array_equal(mask, [False, True, False, True])

    def test_support_mask_uses_original_string_label_mapping(self):
        model = PegasosSVM()
        model._to_pm1(["standard", "premium"])
        model.w = np.array([1.0, 0.0])
        model.b = 0.0
        X = np.array([[2.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [-0.5, 0.0]])
        y = np.array(["standard", "standard", "premium", "premium"])

        mask = model.support_mask(X, y)

        np.testing.assert_array_equal(mask, [False, True, False, True])

    def test_support_mask_rejects_invalid_labels_dimensions_and_sample_count(self):
        model = PegasosSVM()
        model._to_pm1([0, 1])
        model.w = np.array([1.0, 0.0])
        invalid_inputs = [
            (np.ones((2, 2)), np.array([0, 2]), "not seen during fit"),
            (np.ones(2), np.array([0, 1]), "two-dimensional"),
            (np.ones((2, 2)), np.array([[0], [1]]), "one-dimensional"),
            (np.ones((3, 2)), np.array([0, 1]), "same number"),
            (np.ones((2, 3)), np.array([0, 1]), "number of features"),
            (np.array([[np.inf, 0.0], [0.0, 0.0]]), np.array([0, 1]), "NaN"),
            (np.ones((2, 2)), np.array([0.0, np.inf]), "NaN"),
        ]

        for X, y, message in invalid_inputs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model.support_mask(X, y)

    def test_unfitted_model_rejects_diagnostics(self):
        model = PegasosSVM()
        diagnostics = [
            model.weight_norm,
            model.margin_width,
            lambda: model.support_mask([[0.0]], [0]),
        ]

        for diagnostic in diagnostics:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(RuntimeError, "fitted"):
                    diagnostic()

    def test_support_mask_does_not_change_label_mapping(self):
        model = PegasosSVM()
        model._to_pm1(["standard", "premium"])
        model.w = np.array([1.0])
        classes_before = model.classes_.copy()
        mapping_before = model.class_to_pm1_.copy()

        model.support_mask([[1.0], [-1.0]], ["standard", "premium"])

        np.testing.assert_array_equal(model.classes_, classes_before)
        self.assertEqual(model.class_to_pm1_, mapping_before)


if __name__ == "__main__":
    unittest.main()
