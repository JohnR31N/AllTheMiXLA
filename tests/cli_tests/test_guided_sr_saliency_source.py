import unittest
from unittest.mock import patch

import torch

from allthemix.cli.train import build_batch_mixer, call_batch_mixer


class _CaptureGuidedSRMixer:
    def __call__(self, images, targets, saliency_maps=None):
        return saliency_maps


def _edge_batch():
    images = torch.zeros(2, 3, 8, 8)
    images[:, :, :, 4:] = 1.0
    targets = torch.arange(2, dtype=torch.long)
    return images, targets


class GuidedSRSaliencySourceTests(unittest.TestCase):
    def test_build_batch_mixer_passes_dataset_stats_to_online_saliency_methods(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        guided = build_batch_mixer(
            {
                "method": "guided_sr",
                "alpha": 1.0,
                "guidedmixup_blur_kernel": 7,
                "guidedmixup_condition": "greedy",
                "mean": mean,
                "std": std,
            }
        )
        saliency = build_batch_mixer(
            {
                "method": "saliencymix",
                "alpha": 1.0,
                "saliency_source": "spectral_residual",
                "guidedmixup_blur_kernel": 7,
                "saliencymix_no_repeat": False,
                "mean": mean,
                "std": std,
            }
        )

        self.assertEqual(guided.saliency_mean, mean)
        self.assertEqual(guided.saliency_std, std)
        self.assertEqual(saliency.saliency_mean, mean)
        self.assertEqual(saliency.saliency_std, std)

    def test_gradient_source_supplies_fast_saliency_maps_without_cache(self):
        images, targets = _edge_batch()

        saliency_maps = call_batch_mixer(
            _CaptureGuidedSRMixer(),
            images,
            targets,
            {},
            {"method": "guided_sr", "saliency_source": "gradient"},
            False,
            None,
            0,
            1,
        )

        self.assertEqual(tuple(saliency_maps.shape), (2, 1, 8, 8))
        self.assertGreater(float(saliency_maps.max()), 0.0)

    def test_gradient_source_denormalizes_images_when_stats_are_available(self):
        unit_images, targets = _edge_batch()
        mean = (0.5, 0.25, 0.1)
        std = (0.2, 0.5, 0.4)
        normalized = (unit_images - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
        captured = {}

        def fake_gradient(images):
            captured["images"] = images.detach().clone()
            return torch.ones(images.size(0), 1, images.size(-2), images.size(-1))

        with patch("allthemix.methods.saliencymix.compute_gradient_saliency_maps", fake_gradient):
            saliency_maps = call_batch_mixer(
                _CaptureGuidedSRMixer(),
                normalized,
                targets,
                {},
                {"method": "guided_sr", "saliency_source": "gradient", "mean": mean, "std": std},
                False,
                None,
                0,
                1,
            )

        torch.testing.assert_close(captured["images"], unit_images)
        self.assertEqual(tuple(saliency_maps.shape), (2, 1, 8, 8))

    def test_spectral_source_defers_to_guided_sr_online_path(self):
        images, targets = _edge_batch()

        saliency_maps = call_batch_mixer(
            _CaptureGuidedSRMixer(),
            images,
            targets,
            {},
            {"method": "guided_sr", "saliency_source": "spectral_residual"},
            False,
            None,
            0,
            1,
        )

        self.assertIsNone(saliency_maps)

    def test_cached_batch_maps_are_preferred(self):
        images, targets = _edge_batch()
        cached_maps = torch.ones(2, 1, 8, 8)

        saliency_maps = call_batch_mixer(
            _CaptureGuidedSRMixer(),
            images,
            targets,
            {"saliency_maps": cached_maps},
            {"method": "guided_sr", "saliency_source": "gradient"},
            False,
            None,
            0,
            1,
        )

        torch.testing.assert_close(saliency_maps, cached_maps)


if __name__ == "__main__":
    unittest.main()
