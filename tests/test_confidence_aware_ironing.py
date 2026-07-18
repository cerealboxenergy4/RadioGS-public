import unittest

import torch

from utils.loss_utils import (counterfactual_layer_target, gradient_dissent_gate,
                              gradient_dissent_observation, single_layer_confidence,
                              single_layer_loss)


class ConfidenceAwareIroningTest(unittest.TestCase):
    def test_none_path_matches_original_uniform_mean(self):
        alpha = torch.ones(1, 1, 3)
        neff = torch.tensor([[[1.0, 2.0, 3.0]]])
        alpha_m2 = alpha.square() / neff
        loss, mean_neff, foreground = single_layer_loss(alpha, alpha_m2)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        self.assertAlmostEqual(mean_neff.item(), 2.0, places=6)
        self.assertAlmostEqual(foreground.item(), 1.0, places=6)

    def test_confidence_is_a_normalized_weight_not_a_global_scale(self):
        alpha = torch.ones(1, 1, 3)
        neff = torch.tensor([[[1.0, 2.0, 3.0]]])
        alpha_m2 = (alpha.square() / neff).requires_grad_()
        confidence = torch.tensor([[[1.0, 2.0, 1.0]]], requires_grad=True)
        loss, _, _ = single_layer_loss(alpha, alpha_m2, confidence=confidence)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        loss.backward()
        self.assertIsNotNone(alpha_m2.grad)
        self.assertIsNone(confidence.grad)

    def test_counterfactual_target_requires_benefit_and_depth_separation(self):
        gt = torch.zeros(3, 1, 3)
        full = torch.zeros_like(gt)
        first = torch.ones_like(gt)
        # Pixel 0: appearance benefit + separated depth -> permit two layers.
        # Pixel 1: appearance benefit but same depth -> keep K=1 (texture-like boundary).
        # Pixel 2: separated depth but no appearance benefit -> keep K=1.
        full[:, :, 2] = 1.0
        alpha = torch.ones(1, 1, 3)
        first_depth = torch.ones(1, 1, 3)
        expected_depth = torch.tensor([[[1.2, 1.0, 1.2]]])
        target, stats = counterfactual_layer_target(
            gt, full, first, alpha, first_depth, expected_depth,
            benefit_tau=0.1, benefit_temperature=1e-3,
            depth_tau=0.05, depth_temperature=1e-3)
        self.assertGreater(target[0, 0, 0].item(), 1.99)
        self.assertLess(target[0, 0, 1].item(), 1.01)
        self.assertLess(target[0, 0, 2].item(), 1.01)
        self.assertAlmostEqual(stats["permit_fraction"].item(), 1.0 / 3.0, places=5)
        self.assertFalse(target.requires_grad)

    def test_hard_counterfactual_target_is_binary(self):
        gt = torch.zeros(3, 1, 2)
        full = torch.zeros_like(gt)
        first = torch.full_like(gt, 0.2)
        alpha = torch.ones(1, 1, 2)
        first_depth = torch.ones(1, 1, 2)
        expected_depth = torch.tensor([[[1.10, 1.01]]])

        target, stats = counterfactual_layer_target(
            gt, full, first, alpha, first_depth, expected_depth,
            benefit_tau=0.01, depth_tau=0.02, gate_mode="hard")

        self.assertEqual(float(target[0, 0, 0]), 2.0)
        self.assertEqual(float(target[0, 0, 1]), 1.0)
        self.assertEqual(float(stats["permit_fraction"]), 0.5)

    def test_counterfactual_target_rejects_unknown_gate_mode(self):
        image = torch.zeros(3, 1, 1)
        scalar = torch.ones(1, 1, 1)
        with self.assertRaisesRegex(ValueError, "gate_mode"):
            counterfactual_layer_target(
                image, image, image, scalar, scalar, scalar, gate_mode="unknown")

    def test_adaptive_target_relaxes_only_selected_pixels(self):
        alpha = torch.ones(1, 1, 2)
        alpha_m2 = torch.full_like(alpha, 0.5)  # N_eff=2 at both pixels
        fixed, _, _ = single_layer_loss(alpha, alpha_m2)
        target = torch.tensor([[[2.0, 1.0]]])
        adaptive, _, _ = single_layer_loss(alpha, alpha_m2, target=target)
        self.assertAlmostEqual(fixed.item(), 1.0, places=6)
        self.assertAlmostEqual(adaptive.item(), 0.5, places=6)

    def test_edge_mode_protects_high_frequency_pixels(self):
        image = torch.zeros(3, 9, 9)
        image[:, :, 5:] = 1.0
        alpha = torch.ones(1, 9, 9)
        confidence, stats = single_layer_confidence(
            image, alpha, mode="edge", edge_tau=0.05, confidence_floor=0.0)
        smooth = confidence[0, 4, 1].item()
        edge = confidence[0, 4, 4].item()
        self.assertLess(edge, smooth)
        self.assertLess(stats["edge"].item(), 1.0)
        self.assertFalse(confidence.requires_grad)

    def test_hybrid_mode_downweights_depth_and_normal_disagreement(self):
        image = torch.zeros(3, 3, 4)
        alpha = torch.ones(1, 3, 4)
        distortion = torch.ones(1, 3, 4)
        distortion[:, :, -1] = 10.0
        rendered_normal = torch.zeros(3, 3, 4)
        surface_normal = torch.zeros(3, 3, 4)
        rendered_normal[2] = 1.0
        surface_normal[2] = 1.0
        surface_normal[2, :, -1] = -1.0
        confidence, stats = single_layer_confidence(
            image, alpha, distortion, rendered_normal, surface_normal,
            mode="hybrid", confidence_floor=0.0)
        self.assertLess(confidence[0, 1, -1].item(), confidence[0, 1, 0].item())
        self.assertIn("dist", stats)
        self.assertIn("normal", stats)

    def test_gradient_dissent_requires_opposing_nontrivial_pressure(self):
        data = torch.tensor([[-2.0], [2.0], [-0.01], [-3.0]])
        iron = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
        visible = torch.tensor([True, True, True, False])
        score, stats = gradient_dissent_observation(
            data, iron, visible, strength_percentile=1.0, min_pressure_ratio=0.25)

        self.assertGreater(float(score[0]), 0.0)
        self.assertEqual(float(score[1]), 0.0)   # same gradient direction
        self.assertEqual(float(score[2]), 0.0)   # opposing but too weak
        self.assertEqual(float(score[3]), 0.0)   # invisible: caller must not update its EMA
        self.assertAlmostEqual(float(stats["conflict_fraction"]), 1.0 / 3.0, places=6)

    def test_gradient_dissent_gate_is_lagged_budgeted_and_observation_gated(self):
        ema = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0])
        observations = torch.full_like(ema, 30.0)
        observations[-1] = 0.0  # highest raw score is ineligible without enough observations
        gate, protected = gradient_dissent_gate(
            ema, observations, tau=0.25, min_gate=0.1,
            max_protected_fraction=0.2, min_observations=20)

        self.assertEqual(int(protected.sum()), 2)
        self.assertTrue(bool(protected[0]))
        self.assertTrue(bool(protected[1]))
        self.assertFalse(bool(protected[-1]))
        self.assertLess(float(gate[0]), 1.0)
        self.assertEqual(float(gate[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
