import unittest

import torch

from allthemix.cli.train import build_batch_mixer, call_batch_mixer, cross_device_shuffle_batch
from allthemix.methods.saliencymix import compute_gradient_saliency_maps


class _FakeXm:
    def __init__(self, world_size=2):
        self.world_size = int(world_size)

    def all_gather(self, tensor, dim=0):
        shards = []
        for shard in range(self.world_size):
            if torch.is_floating_point(tensor):
                offset = 1000.0 if tensor.dim() == 1 else 100.0
                shards.append(tensor + shard * offset)
            else:
                shards.append(tensor + shard * tensor.size(0))
        return torch.cat(shards, dim=dim)


class _ExplicitXm:
    def __init__(
        self,
        global_images: torch.Tensor,
        global_targets: torch.Tensor,
        global_scores: torch.Tensor,
        global_aux: torch.Tensor | None = None,
    ) -> None:
        self.global_images = global_images
        self.global_targets = global_targets
        self.global_scores = global_scores
        self.global_aux = global_aux

    def all_gather(self, tensor, dim=0):
        self.assert_dim(dim)
        if tensor.dim() == 1 and torch.is_floating_point(tensor):
            return self.global_scores.to(device=tensor.device)
        if tensor.dim() == 1:
            return self.global_targets.to(device=tensor.device)
        if tensor.dim() == 4 and tensor.size(1) == self.global_images.size(1):
            return self.global_images.to(device=tensor.device)
        if self.global_aux is None:
            raise AssertionError("unexpected auxiliary all_gather call")
        return self.global_aux.to(device=tensor.device)

    @staticmethod
    def assert_dim(dim: int) -> None:
        if dim != 0:
            raise AssertionError(f"expected all_gather(dim=0), got dim={dim}")


class _RecordingMixer:
    def __call__(self, images, targets, partner_images=None, partner_targets=None, index=None, **kwargs):
        del images, targets, partner_images, partner_targets
        return kwargs.get("index", index)


class _RecordingSaliencyMixer:
    def __call__(
        self,
        images,
        targets,
        partner_images=None,
        partner_targets=None,
        partner_saliency_maps=None,
        index=None,
        **kwargs,
    ):
        del images, targets, partner_images, kwargs
        return index, partner_targets, partner_saliency_maps


class _RecordingGuidedSRMixer:
    def __call__(
        self,
        images,
        targets,
        saliency_maps=None,
        partner_images=None,
        partner_targets=None,
        partner_saliency_maps=None,
        index=None,
        **kwargs,
    ):
        del images, targets, saliency_maps, kwargs
        return index, partner_images, partner_targets, partner_saliency_maps


class _RecordingGuidedSRMixerWithLocalSaliency:
    def __call__(
        self,
        images,
        targets,
        saliency_maps=None,
        partner_images=None,
        partner_targets=None,
        partner_saliency_maps=None,
        index=None,
        **kwargs,
    ):
        del images, targets, partner_images, partner_targets, kwargs
        return saliency_maps, partner_saliency_maps, index


class _RecordingCatchUpMixer:
    def __call__(self, images, targets, partner_images=None, partner_targets=None, index=None, **kwargs):
        del images, targets, kwargs
        return partner_images, partner_targets, index


class CrossDeviceShuffleTests(unittest.TestCase):
    def test_cross_device_shuffle_uses_global_no_repeat_mapping_for_each_rank(self):
        local_batch = 2
        world_size = 3
        total_batch = local_batch * world_size
        global_images = torch.arange(total_batch * 3, dtype=torch.float32).view(total_batch, 3, 1, 1)
        global_targets = torch.arange(100, 100 + total_batch)
        global_aux = torch.arange(total_batch, dtype=torch.float32).view(total_batch, 1, 1, 1) + 0.5
        global_scores = torch.tensor([0.60, 0.10, 0.30, 0.90, 0.20, 0.50])

        order = torch.argsort(global_scores)
        expected_mapping = torch.empty_like(order)
        expected_mapping.scatter_(0, order, torch.roll(order, shifts=1, dims=0))

        for rank in range(world_size):
            with self.subTest(rank=rank):
                start = rank * local_batch
                local_images = global_images.narrow(0, start, local_batch)
                local_targets = global_targets.narrow(0, start, local_batch)
                local_aux = global_aux.narrow(0, start, local_batch)

                partner_images, partner_targets, partner_index, partner_aux = cross_device_shuffle_batch(
                    local_images,
                    local_targets,
                    rank,
                    _ExplicitXm(global_images, global_targets, global_scores, global_aux),
                    aux_tensor=local_aux,
                    no_repeat=True,
                )

                expected_index = expected_mapping.narrow(0, start, local_batch)
                local_global_index = torch.arange(start, start + local_batch)
                torch.testing.assert_close(partner_index.cpu(), expected_index)
                self.assertTrue(torch.all(partner_index.cpu() != local_global_index))
                torch.testing.assert_close(partner_images.cpu(), global_images.index_select(0, expected_index))
                torch.testing.assert_close(partner_targets.cpu(), global_targets.index_select(0, expected_index))
                torch.testing.assert_close(partner_aux.cpu(), global_aux.index_select(0, expected_index))

    def test_cross_device_shuffle_rejects_world_size_mismatch(self):
        images = torch.arange(2, dtype=torch.float32).view(2, 1, 1, 1)
        targets = torch.arange(2)

        with self.assertRaisesRegex(ValueError, "world size mismatch"):
            cross_device_shuffle_batch(
                images,
                targets,
                rank=0,
                xm=_FakeXm(world_size=2),
                world_size=3,
            )

    def test_cross_device_shuffle_rejects_rank_outside_gathered_world_size(self):
        images = torch.arange(2, dtype=torch.float32).view(2, 1, 1, 1)
        targets = torch.arange(2)

        with self.assertRaisesRegex(ValueError, "rank is outside gathered world size"):
            cross_device_shuffle_batch(
                images,
                targets,
                rank=2,
                xm=_FakeXm(world_size=2),
                world_size=2,
            )

    def test_openmixup_dist_methods_use_no_repeat_partner_indices(self):
        local_batch = 4
        images = torch.arange(local_batch, dtype=torch.float32).view(local_batch, 1, 1, 1)
        targets = torch.arange(local_batch)
        saliency_maps = torch.zeros(local_batch, 1, 1, 1)

        for method, no_repeat_field in (
            ("mixup", "mixup_no_repeat"),
            ("cutmix", "cutmix_no_repeat"),
            ("resizemix", "resizemix_no_repeat"),
            ("fmix", "fmix_no_repeat"),
            ("saliencymix", "saliencymix_no_repeat"),
        ):
            for rank in (0, 1):
                with self.subTest(method=method, rank=rank):
                    aux_info = {"saliency_maps": saliency_maps} if method == "saliencymix" else {}
                    index = call_batch_mixer(
                        _RecordingMixer(),
                        images,
                        targets,
                        aux_info,
                        {
                            "method": method,
                            "cross_device_shuffle": True,
                            no_repeat_field: True,
                        },
                        use_xla=True,
                        xm=_FakeXm(world_size=2),
                        rank=rank,
                        world_size=2,
                    )

                    self.assertEqual(index.numel(), local_batch)
                    self.assertTrue(torch.all((index.cpu() >= 0) & (index.cpu() < 2 * local_batch)))
                    local_global_index = torch.arange(local_batch) + rank * local_batch
                    self.assertTrue(torch.all(index.cpu() != local_global_index))

    def test_xla_cross_device_path_uses_real_mixer_external_partners(self):
        local_batch = 4
        images = torch.arange(local_batch * 3 * 8 * 8, dtype=torch.float32).view(local_batch, 3, 8, 8)
        targets = torch.arange(local_batch)
        global_targets = torch.arange(local_batch * 2)
        common_config = {
            "alpha": 1.0,
            "image_size": 8,
            "cross_device_shuffle": True,
            "mixup_no_repeat": False,
            "cutmix_no_repeat": False,
            "resizemix_no_repeat": False,
            "fmix_no_repeat": False,
            "resizemix_scope_min": 0.1,
            "resizemix_scope_max": 0.4,
            "resizemix_use_alpha": False,
            "decay_power": 3.0,
            "max_soft": 0.0,
            "reformulate": False,
        }

        for method in ("mixup", "cutmix", "resizemix", "fmix"):
            with self.subTest(method=method):
                torch.manual_seed(0)
                result = call_batch_mixer(
                    build_batch_mixer({**common_config, "method": method}),
                    images,
                    targets,
                    {},
                    {**common_config, "method": method},
                    use_xla=True,
                    xm=_FakeXm(world_size=2),
                    rank=1,
                    world_size=2,
                )

                self.assertEqual(result.images.shape, images.shape)
                self.assertEqual(result.index.numel(), local_batch)
                self.assertTrue(torch.all((result.index.cpu() >= 0) & (result.index.cpu() < 2 * local_batch)))
                torch.testing.assert_close(result.targets_a.cpu(), targets)
                torch.testing.assert_close(result.targets_b.cpu(), global_targets.index_select(0, result.index.cpu()))

    def test_saliencymix_cross_device_shuffle_keeps_partner_saliency_aligned(self):
        local_batch = 4
        images = torch.arange(local_batch, dtype=torch.float32).view(local_batch, 1, 1, 1)
        targets = torch.arange(local_batch)
        saliency_maps = torch.arange(local_batch, dtype=torch.float32).view(local_batch, 1, 1, 1)
        rank = 1

        index, partner_targets, partner_saliency_maps = call_batch_mixer(
            _RecordingSaliencyMixer(),
            images,
            targets,
            {"saliency_maps": saliency_maps},
            {
                "method": "saliencymix",
                "cross_device_shuffle": True,
                "saliencymix_no_repeat": False,
            },
            use_xla=True,
            xm=_FakeXm(world_size=2),
            rank=rank,
            world_size=2,
        )

        global_targets = torch.arange(local_batch * 2)
        global_saliency_maps = torch.cat([saliency_maps, saliency_maps + 100.0], dim=0)
        torch.testing.assert_close(partner_targets.cpu(), global_targets.index_select(0, index.cpu()))
        torch.testing.assert_close(partner_saliency_maps.cpu(), global_saliency_maps.index_select(0, index.cpu()))

    def test_guided_sr_random_cross_device_shuffle_keeps_partner_saliency_aligned(self):
        local_batch = 4
        images = torch.arange(local_batch, dtype=torch.float32).view(local_batch, 1, 1, 1)
        targets = torch.arange(local_batch)
        saliency_maps = torch.arange(local_batch, dtype=torch.float32).view(local_batch, 1, 1, 1)
        rank = 1

        index, partner_images, partner_targets, partner_saliency_maps = call_batch_mixer(
            _RecordingGuidedSRMixer(),
            images,
            targets,
            {"saliency_maps": saliency_maps},
            {
                "method": "guided_sr",
                "cross_device_shuffle": True,
                "guidedmixup_condition": "random",
                "saliency_source": "batch",
            },
            use_xla=True,
            xm=_FakeXm(world_size=2),
            rank=rank,
            world_size=2,
        )

        global_images = torch.cat([images, images + 100.0], dim=0)
        global_targets = torch.arange(local_batch * 2)
        global_saliency_maps = torch.cat([saliency_maps, saliency_maps + 100.0], dim=0)
        torch.testing.assert_close(partner_images.cpu(), global_images.index_select(0, index.cpu()))
        torch.testing.assert_close(partner_targets.cpu(), global_targets.index_select(0, index.cpu()))
        torch.testing.assert_close(partner_saliency_maps.cpu(), global_saliency_maps.index_select(0, index.cpu()))

    def test_guided_sr_gradient_source_cross_device_shuffle_gathers_fast_saliency_maps(self):
        local_batch = 4
        images = torch.zeros(local_batch, 3, 4, 4)
        for item in range(local_batch):
            images[item, :, :, item % 4 :] = 1.0
        targets = torch.arange(local_batch)
        rank = 1

        saliency_maps, partner_saliency_maps, index = call_batch_mixer(
            _RecordingGuidedSRMixerWithLocalSaliency(),
            images,
            targets,
            {},
            {
                "method": "guided_sr",
                "cross_device_shuffle": True,
                "guidedmixup_condition": "random",
                "saliency_source": "gradient",
            },
            use_xla=True,
            xm=_FakeXm(world_size=2),
            rank=rank,
            world_size=2,
        )

        expected_local_saliency = compute_gradient_saliency_maps(images)
        expected_global_saliency = torch.cat(
            [expected_local_saliency, expected_local_saliency + 100.0],
            dim=0,
        )
        torch.testing.assert_close(saliency_maps.cpu(), expected_local_saliency)
        torch.testing.assert_close(partner_saliency_maps.cpu(), expected_global_saliency.index_select(0, index.cpu()))

    def test_catchupmix_ignores_cross_device_shuffle_to_keep_feature_hook_local(self):
        images = torch.randn(4, 3, 8, 8)
        targets = torch.arange(4)

        partner_images, partner_targets, index = call_batch_mixer(
            _RecordingCatchUpMixer(),
            images,
            targets,
            {},
            {
                "method": "catchupmix",
                "cross_device_shuffle": True,
                "catchupmix_no_repeat": False,
            },
            use_xla=True,
            xm=_FakeXm(world_size=2),
            rank=1,
            world_size=2,
        )

        self.assertIsNone(partner_images)
        self.assertIsNone(partner_targets)
        self.assertIsNone(index)


if __name__ == "__main__":
    unittest.main()
