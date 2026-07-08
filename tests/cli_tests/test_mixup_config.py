import argparse
from contextlib import redirect_stderr
from io import StringIO
import random
import unittest
from unittest.mock import patch

from allthemix.cli.train import (
    build_batch_mixer,
    load_config,
    normalize_method_name,
    parse_args,
    resolved_config,
    should_apply_method,
    training_needs_batch_saliency_maps,
    validate_global_batch_size,
    validate_resolved_config,
)


def _args(**overrides):
    defaults = {
        "dataset": None,
        "recipe": None,
        "method": None,
        "data_dir": None,
        "output_dir": None,
        "checkpoint_dir": None,
        "download": None,
        "no_augment": None,
        "epochs": None,
        "batch_size": None,
        "lr": None,
        "momentum": None,
        "weight_decay": None,
        "scheduler": None,
        "milestones": None,
        "alpha": None,
        "decay_power": None,
        "max_soft": None,
        "reformulate": None,
        "fmix_prob": None,
        "mix_prob": None,
        "guidedmixup_blur_kernel": None,
        "guidedmixup_condition": None,
        "saliency_source": None,
        "saliency_dir": None,
        "saliency_path": None,
        "aug_recipe": None,
        "sal_aug_recipe": None,
        "checkpoint": None,
        "final_test_checkpoint": None,
        "eval_only": False,
        "seed": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class MixUpConfigTests(unittest.TestCase):
    def test_method_aliases_cover_allthemix_style_names(self):
        expected = {
            "mix_up": "mixup",
            "mix-up": "mixup",
            "cut_mix": "cutmix",
            "cut-mix": "cutmix",
            "f_mix": "fmix",
            "f-mix": "fmix",
            "guidedmixup": "guided_sr",
            "guided_mixup": "guided_sr",
            "guided-mixup": "guided_sr",
            "resize": "resizemix",
            "resize_mix": "resizemix",
            "catch_up_mix": "catchupmix",
            "catch-up-mix": "catchupmix",
            "saliency_mix": "saliencymix",
        }

        for alias, method in expected.items():
            with self.subTest(alias=alias):
                self.assertEqual(normalize_method_name(alias), method)

    def test_parse_args_accepts_allthemix_style_aliases(self):
        args = parse_args(
            [
                "--dataset",
                "tiny_imagenet",
                "--method",
                "guidedmixup",
                "--learning-rate",
                "0.1",
                "--lr-schedule",
                "step",
                "--lr-decay-epochs",
                "150",
                "180",
                "--max-eval-steps",
                "5",
                "--resume-checkpoint",
                "best.pt",
                "--final-test-checkpoint",
                "best",
                "--eval-only",
            ]
        )

        self.assertEqual(args.dataset, "tiny_imagenet")
        self.assertEqual(args.method, "guidedmixup")
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.scheduler, "step")
        self.assertEqual(args.milestones, [150, 180])
        self.assertEqual(args.max_val_steps, 5)
        self.assertEqual(args.checkpoint, "best.pt")
        self.assertEqual(args.final_test_checkpoint, "best")
        self.assertTrue(args.eval_only)

    def test_parse_args_rejects_negative_runtime_intervals(self):
        for args in [
            ["--log-interval", "-1"],
            ["--save-every", "-1"],
            ["--num-workers", "-1"],
        ]:
            with self.subTest(args=args), self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                parse_args(args)

    def test_final_test_checkpoint_resolves_from_config_and_cli(self):
        config = resolved_config(
            _args(final_test_checkpoint="best"),
            {
                "dataset": "tiny_imagenet",
                "method": "baseline",
                "final_test": True,
                "final_test_checkpoint": "last",
            },
        )

        self.assertTrue(config["final_test"])
        self.assertEqual(config["final_test_checkpoint"], "best")

    def test_eval_only_does_not_need_training_saliency_cache(self):
        config = {
            "method": "saliencymix",
            "saliency_source": "batch",
            "epochs": 200,
        }

        self.assertTrue(training_needs_batch_saliency_maps(config, _args()))
        self.assertFalse(training_needs_batch_saliency_maps(config, _args(eval_only=True)))
        self.assertFalse(training_needs_batch_saliency_maps({**config, "epochs": 0}, _args()))
        self.assertFalse(
            training_needs_batch_saliency_maps(
                {**config, "saliency_source": "gradient"},
                _args(),
            )
        )

    def test_mixup_config_resolves_method_section(self):
        raw_config = load_config("configs/cifar10/preact_resnet18/mixup.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["method"], "mixup")
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["method_prob"], 1.0)
        self.assertTrue(config["cross_device_shuffle"])
        self.assertEqual(config["output_dir"], "./runs/mixup")

    def test_cli_method_overrides_config(self):
        raw_config = load_config("configs/cifar10/preact_resnet18/fmix.yaml")
        config = resolved_config(_args(method="mixup", alpha=0.4, mix_prob=0.5), raw_config)

        self.assertEqual(config["method"], "mixup")
        self.assertEqual(config["alpha"], 0.4)
        self.assertEqual(config["method_prob"], 0.5)

    def test_baseline_config_resolves_without_mix_method(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/baseline.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["method"], "baseline")
        self.assertEqual(config["output_dir"], "./runs/baseline")
        self.assertEqual(config["dataset"], "tinyimagenet")
        self.assertEqual(config["model_impl_version"], 2)

    def test_resolved_config_stamps_model_impl_version(self):
        preact_config = resolved_config(
            _args(),
            {"dataset": "tiny_imagenet", "method": "baseline", "model": "preact-resnet18"},
        )
        torch_config = resolved_config(
            _args(),
            {"dataset": "imagenet_a", "method": "baseline", "model": "torchvision_resnet101"},
        )

        self.assertEqual(preact_config["model"], "preact_resnet18")
        self.assertEqual(torch_config["model"], "torch_resnet101")
        self.assertEqual(preact_config["model_impl_version"], 2)
        self.assertEqual(torch_config["model_impl_version"], 1)

    def test_legacy_tiny_baseline_config_resolves_old_field_names(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/baseline_legacy.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["dataset"], "tinyimagenet")
        self.assertEqual(config["method"], "baseline")
        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["epochs"], 200)
        self.assertEqual(config["lr"], 0.1)
        self.assertEqual(config["weight_decay"], 0.0005)
        self.assertEqual(config["scheduler"], "multistep")
        self.assertEqual(config["milestones"], [150, 180])
        self.assertEqual(config["validation_split"], 0.1)
        self.assertTrue(config["final_test"])
        self.assertEqual(config["run_name"], "tiny_imagenet_preact_resnet18_baseline")

    def test_legacy_tiny_fmix_config_matches_baseline_schedule(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/fmix_legacy.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["dataset"], "tinyimagenet")
        self.assertEqual(config["method"], "fmix")
        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["epochs"], 200)
        self.assertEqual(config["lr"], 0.1)
        self.assertEqual(config["weight_decay"], 0.0005)
        self.assertEqual(config["scheduler"], "multistep")
        self.assertEqual(config["milestones"], [150, 180])
        self.assertEqual(config["alpha"], 1.0)
        self.assertTrue(config["cross_device_shuffle"])
        self.assertEqual(config["run_name"], "tiny_imagenet_preact_resnet18_fmix")

    def test_fmix_legacy_flat_decay_field_resolves(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "fmix",
                "fmix_alpha": 1.0,
                "fmix_decay": 2.5,
                "fmix_max_soft": 0.1,
            },
        )

        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["decay_power"], 2.5)
        self.assertEqual(config["max_soft"], 0.1)

    def test_resolved_config_rejects_invalid_core_training_values(self):
        invalid_cases = [
            ({"dataset": "tiny_imagenet", "method": "baseline", "batch_size": 0}, "batch_size"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "epochs": -1}, "epochs"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "learning_rate": -0.1}, "lr"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "learning_rate": float("nan")}, "finite"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "lr_decay_rate": float("inf")}, "finite"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "lr_decay_epochs": [-1]}, "milestones"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "lr_decay_epochs": "150,180"}, "milestones"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "validation_split": 1.0}, "validation_split"),
            ({"dataset": "tiny_imagenet", "method": "mixup", "mixup_prob": 1.5}, "method_prob"),
            ({"dataset": "tiny_imagenet", "method": "mixup", "mixup_prob": -0.1}, "method_prob"),
            ({"dataset": "tiny_imagenet", "method": "baseline", "global_batch_size": 0}, "global_batch_size"),
        ]

        for raw_config, pattern in invalid_cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    resolved_config(_args(), raw_config)

    def test_validate_resolved_config_accepts_zero_epoch_eval_debug_config(self):
        validate_resolved_config(
            {
                "method": "baseline",
                "epochs": 0,
                "batch_size": 32,
                "num_classes": 200,
                "image_size": 64,
                "lr": 0.0,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "lr_decay_rate": 0.0,
                "min_learning_rate": 0.0,
                "milestones": [],
                "method_prob": 0.0,
                "validation_split": 0.0,
                "global_batch_size": None,
            }
        )

    def test_resolved_config_rejects_invalid_method_specific_values(self):
        invalid_cases = [
            ({"dataset": "tiny_imagenet", "method": "cutmix", "cutmix_alpha": 0.0}, "alpha"),
            ({"dataset": "tiny_imagenet", "method": "fmix", "fmix_decay_power": 0.0}, "decay_power"),
            ({"dataset": "tiny_imagenet", "method": "fmix", "fmix_max_soft": -0.1}, "max_soft"),
            ({"dataset": "tiny_imagenet", "method": "resizemix", "resizemix_scope_min": 0.0}, "ResizeMix scope"),
            ({"dataset": "tiny_imagenet", "method": "catchupmix", "catchupmix_cutmix_alpha": 0.0}, "cutmix_alpha"),
            ({"dataset": "tiny_imagenet", "method": "catchupmix", "catchupmix_num_layers": 0}, "num_layers"),
            ({"dataset": "tiny_imagenet", "method": "guided_sr", "guidedmixup_blur_kernel": 4}, "blur_kernel"),
            ({"dataset": "tiny_imagenet", "method": "guided_sr", "guidedmixup_condition": "bad"}, "condition"),
            ({"dataset": "tiny_imagenet", "method": "saliencymix", "saliency_source": "bad"}, "saliency_source"),
        ]

        for raw_config, pattern in invalid_cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    resolved_config(_args(), raw_config)

    def test_mixup_and_fmix_alpha_zero_remain_explicit_no_mix_debug_values(self):
        mixup_config = resolved_config(
            _args(),
            {"dataset": "tiny_imagenet", "method": "mixup", "mixup_alpha": 0.0},
        )
        fmix_config = resolved_config(
            _args(),
            {"dataset": "tiny_imagenet", "method": "fmix", "fmix_alpha": 0.0},
        )

        self.assertEqual(mixup_config["alpha"], 0.0)
        self.assertEqual(fmix_config["alpha"], 0.0)

    def test_paper_tiny_fmix_xla4_config_uses_official_recipe_with_global_batch_128(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/fmix_paper_xla4.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["dataset"], "tinyimagenet")
        self.assertEqual(config["method"], "fmix")
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["epochs"], 200)
        self.assertEqual(config["lr"], 0.1)
        self.assertEqual(config["weight_decay"], 0.0001)
        self.assertEqual(config["scheduler"], "multistep")
        self.assertEqual(config["milestones"], [150, 180])
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["decay_power"], 3.0)
        self.assertEqual(config["validation_split"], 0.0)
        self.assertFalse(config["final_test"])
        self.assertTrue(config["cross_device_shuffle"])

    def test_paper_tiny_cutmix_xla4_config_matches_fmix_paper_protocol(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/cutmix_paper_xla4.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["dataset"], "tinyimagenet")
        self.assertEqual(config["method"], "cutmix")
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["global_batch_size"], 128)
        self.assertEqual(config["epochs"], 200)
        self.assertEqual(config["lr"], 0.1)
        self.assertEqual(config["weight_decay"], 0.0001)
        self.assertEqual(config["scheduler"], "multistep")
        self.assertEqual(config["milestones"], [150, 180])
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["cutmix_prob"], 1.0)
        self.assertTrue(config["cutmix_no_repeat"])
        self.assertTrue(config["use_basic_augmentation"])
        self.assertEqual(config["validation_split"], 0.0)
        self.assertTrue(config["eval_on_test_each_epoch"])
        self.assertFalse(config["final_test"])
        self.assertTrue(config["cross_device_shuffle"])

    def test_table_tiny_xla4_configs_share_allthemix_schedule(self):
        expected_probs = {
            "baseline": 1.0,
            "mixup": 1.0,
            "fmix": 1.0,
            "cutmix": 1.0,
            "resizemix": 1.0,
            "guided_sr": 0.5,
            "saliencymix": 0.5,
            "catchupmix": 1.0,
        }
        for method, expected_prob in expected_probs.items():
            with self.subTest(method=method):
                raw_config = load_config(f"configs/tiny_imagenet/preact_resnet18/{method}_xla4.yaml")
                config = resolved_config(_args(), raw_config)

                self.assertEqual(config["dataset"], "tinyimagenet")
                self.assertEqual(config["method"], method)
                self.assertEqual(config["batch_size"], 32)
                self.assertEqual(config["global_batch_size"], 128)
                self.assertFalse(config["use_basic_augmentation"])
                if method == "saliencymix":
                    self.assertEqual(config["sal_aug_recipe"], "tiny_openmixup")
                else:
                    self.assertEqual(config["aug_recipe"], "tiny_openmixup")
                self.assertEqual(config["epochs"], 200)
                self.assertEqual(config["lr"], 0.1)
                self.assertEqual(config["weight_decay"], 0.0005)
                self.assertEqual(config["scheduler"], "multistep")
                self.assertEqual(config["milestones"], [150, 180])
                self.assertEqual(config["validation_split"], 0.1)
                self.assertTrue(config["final_test"])
                self.assertEqual(config["final_test_checkpoint"], "best")
                self.assertTrue(config["run_metadata_required"])
                self.assertEqual(config["output_dir"], "./outputs")
                self.assertEqual(config["method_prob"], expected_prob)
                if method == "mixup":
                    self.assertFalse(config["mixup_no_repeat"])
                if method == "fmix":
                    self.assertFalse(config["fmix_no_repeat"])
                if method == "resizemix":
                    self.assertEqual(config["resizemix_scope_min"], 0.1)
                    self.assertEqual(config["resizemix_scope_max"], 0.4)
                    self.assertFalse(config["resizemix_use_alpha"])
                    self.assertFalse(config["resizemix_no_repeat"])
                if method == "saliencymix":
                    self.assertFalse(config["saliencymix_no_repeat"])

    def test_cli_checkpoint_dir_overrides_config(self):
        config = resolved_config(
            _args(checkpoint_dir="/tmp/debug_checkpoints"),
            {
                "dataset": "tiny_imagenet",
                "method": "baseline",
                "checkpoint_dir": "./checkpoints",
            },
        )

        self.assertEqual(config["checkpoint_dir"], "/tmp/debug_checkpoints")

    def test_guided_sr_legacy_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "guidedmixup",
                "guidedmixup_alpha": 1.0,
                "guidedmixup_prob": 0.5,
                "guidedmixup_blur_kernel": 7,
                "guidedmixup_condition": "greedy",
            },
        )

        self.assertEqual(config["method"], "guided_sr")
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["method_prob"], 0.5)
        self.assertEqual(config["guidedmixup_blur_kernel"], 7)
        self.assertEqual(config["guidedmixup_condition"], "greedy")

    def test_guided_sr_alias_section_resolves(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "guidedmixup",
                "guidedmixup": {
                    "alpha": 0.5,
                    "prob": 0.25,
                    "blur_kernel": 9,
                    "condition": "random",
                    "saliency_source": "batch",
                },
            },
        )

        self.assertEqual(config["method"], "guided_sr")
        self.assertEqual(config["alpha"], 0.5)
        self.assertEqual(config["method_prob"], 0.25)
        self.assertEqual(config["guidedmixup_blur_kernel"], 9)
        self.assertEqual(config["guidedmixup_condition"], "random")
        self.assertEqual(config["saliency_source"], "batch")

    def test_guided_sr_alias_flat_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "guided-sr",
                "guided_sr_alpha": 0.6,
                "guided_sr_prob": 0.75,
                "guided_sr_blur_kernel": 11,
                "guided_sr_condition": "random",
                "guided_sr_saliency_source": "batch",
            },
        )

        self.assertEqual(config["method"], "guided_sr")
        self.assertEqual(config["alpha"], 0.6)
        self.assertEqual(config["method_prob"], 0.75)
        self.assertEqual(config["guidedmixup_blur_kernel"], 11)
        self.assertEqual(config["guidedmixup_condition"], "random")
        self.assertEqual(config["saliency_source"], "batch")

    def test_cutmix_legacy_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "cutmix",
                "cutmix_alpha": 1.0,
                "cutmix_prob": 1.0,
                "cutmix_no_repeat": True,
            },
        )

        self.assertEqual(config["method"], "cutmix")
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["method_prob"], 1.0)
        self.assertTrue(config["cutmix_no_repeat"])

    def test_cutmix_alias_section_resolves(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "cut-mix",
                "cut-mix": {
                    "alpha": 1.2,
                    "prob": 0.5,
                    "no_repeat": True,
                },
            },
        )

        self.assertEqual(config["method"], "cutmix")
        self.assertEqual(config["alpha"], 1.2)
        self.assertEqual(config["method_prob"], 0.5)
        self.assertTrue(config["cutmix_no_repeat"])

    def test_saliencymix_no_repeat_reaches_batch_mixer(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "saliencymix",
                "saliencymix_alpha": 1.0,
                "saliencymix_no_repeat": True,
            },
        )

        mixer = build_batch_mixer(config)

        self.assertTrue(mixer.no_repeat)

    def test_mixup_no_repeat_reaches_batch_mixer(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "mixup",
                "mixup_no_repeat": True,
            },
        )

        mixer = build_batch_mixer(config)

        self.assertTrue(mixer.no_repeat)

    def test_fmix_no_repeat_reaches_batch_mixer(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "fmix",
                "fmix": {"no_repeat": True},
            },
        )

        mixer = build_batch_mixer(config)

        self.assertTrue(mixer.no_repeat)

    def test_resizemix_use_alpha_reaches_batch_mixer(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "resizemix",
                "resizemix_alpha": 0.7,
                "resizemix_use_alpha": True,
            },
        )

        mixer = build_batch_mixer(config)

        self.assertEqual(mixer.alpha, 0.7)
        self.assertTrue(mixer.use_alpha)

    def test_catchupmix_legacy_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "catch_up_mix",
                "catchupmix_alpha": 1.0,
                "catchupmix_cutmix_alpha": 1.0,
                "catchupmix_num_layers": 5,
                "catchupmix_no_repeat": False,
            },
        )

        self.assertEqual(config["method"], "catchupmix")
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["catchupmix_cutmix_alpha"], 1.0)
        self.assertEqual(config["catchupmix_num_layers"], 5)
        self.assertFalse(config["catchupmix_no_repeat"])

    def test_catchupmix_alias_section_resolves(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "catch_up_mix",
                "catch_up_mix": {
                    "alpha": 1.1,
                    "prob": 0.6,
                    "cutmix_alpha": 0.7,
                    "num_layers": 4,
                    "no_repeat": True,
                },
            },
        )

        self.assertEqual(config["method"], "catchupmix")
        self.assertEqual(config["alpha"], 1.1)
        self.assertEqual(config["method_prob"], 0.6)
        self.assertEqual(config["catchupmix_cutmix_alpha"], 0.7)
        self.assertEqual(config["catchupmix_num_layers"], 4)
        self.assertTrue(config["catchupmix_no_repeat"])

    def test_catchupmix_alias_flat_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "catch-up-mix",
                "catch_up_mix_alpha": 1.2,
                "catch_up_mix_prob": 0.8,
                "catch_up_mix_cutmix_alpha": 0.9,
                "catch_up_mix_num_layers": 3,
                "catch_up_mix_no_repeat": True,
            },
        )

        self.assertEqual(config["method"], "catchupmix")
        self.assertEqual(config["alpha"], 1.2)
        self.assertEqual(config["method_prob"], 0.8)
        self.assertEqual(config["catchupmix_cutmix_alpha"], 0.9)
        self.assertEqual(config["catchupmix_num_layers"], 3)
        self.assertTrue(config["catchupmix_no_repeat"])

    def test_resizemix_legacy_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "resize",
                "resizemix_scope_min": 0.1,
                "resizemix_scope_max": 0.4,
                "resizemix_use_alpha": True,
                "resizemix_prob": 1.0,
                "resizemix_no_repeat": True,
            },
        )

        self.assertEqual(config["method"], "resizemix")
        self.assertEqual(config["method_prob"], 1.0)
        self.assertEqual(config["resizemix_scope_min"], 0.1)
        self.assertEqual(config["resizemix_scope_max"], 0.4)
        self.assertTrue(config["resizemix_use_alpha"])
        self.assertTrue(config["resizemix_no_repeat"])

    def test_resizemix_alias_section_resolves(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "resize_mix",
                "resize_mix": {
                    "alpha": 0.7,
                    "prob": 0.6,
                    "scope_min": 0.2,
                    "scope_max": 0.7,
                    "use_alpha": True,
                    "no_repeat": True,
                },
            },
        )

        self.assertEqual(config["method"], "resizemix")
        self.assertEqual(config["alpha"], 0.7)
        self.assertEqual(config["method_prob"], 0.6)
        self.assertEqual(config["resizemix_scope_min"], 0.2)
        self.assertEqual(config["resizemix_scope_max"], 0.7)
        self.assertTrue(config["resizemix_use_alpha"])
        self.assertTrue(config["resizemix_no_repeat"])

    def test_resizemix_alias_flat_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "resize-mix",
                "resize_mix_alpha": 0.8,
                "resize_mix_prob": 0.5,
                "resize_mix_scope_min": 0.2,
                "resize_mix_scope_max": 0.7,
                "resize_mix_use_alpha": True,
                "resize_mix_no_repeat": True,
            },
        )

        self.assertEqual(config["method"], "resizemix")
        self.assertEqual(config["alpha"], 0.8)
        self.assertEqual(config["method_prob"], 0.5)
        self.assertEqual(config["resizemix_scope_min"], 0.2)
        self.assertEqual(config["resizemix_scope_max"], 0.7)
        self.assertTrue(config["resizemix_use_alpha"])
        self.assertTrue(config["resizemix_no_repeat"])

    def test_method_section_generic_fields_do_not_pollute_unrelated_metadata(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "resize_mix",
                "resize_mix": {
                    "alpha": 0.7,
                    "prob": 0.6,
                    "scope_min": 0.2,
                    "scope_max": 0.7,
                    "use_alpha": True,
                    "no_repeat": True,
                },
            },
        )

        self.assertEqual(config["method"], "resizemix")
        self.assertTrue(config["resizemix_no_repeat"])
        self.assertFalse(config["mixup_no_repeat"])
        self.assertFalse(config["fmix_no_repeat"])
        self.assertFalse(config["cutmix_no_repeat"])
        self.assertFalse(config["catchupmix_no_repeat"])
        self.assertFalse(config["saliencymix_no_repeat"])

    def test_irrelevant_method_flat_fields_do_not_change_baseline_metadata(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "baseline",
                "method_prob": 0.2,
                "mixup_prob": 0.2,
                "mix_alpha": 0.3,
                "mixup_no_repeat": True,
                "decay_power": 2.0,
                "max_soft": 0.2,
                "reformulate": True,
                "fmix": {"alpha": 0.4, "no_repeat": True, "decay_power": 2.0, "max_soft": 0.2},
                "saliency_source": "batch",
            },
        )

        self.assertEqual(config["method"], "baseline")
        self.assertEqual(config["method_prob"], 1.0)
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["decay_power"], 3.0)
        self.assertEqual(config["max_soft"], 0.0)
        self.assertFalse(config["reformulate"])
        self.assertFalse(config["mixup_no_repeat"])
        self.assertFalse(config["fmix_no_repeat"])
        self.assertEqual(config["saliency_source"], "spectral_residual")

    def test_saliencymix_legacy_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "saliency_mix",
                "saliencymix_alpha": 1.0,
                "saliencymix_prob": 0.5,
                "saliency_source": "batch",
                "basic_aug": False,
                "sal_aug_recipe": "tiny_openmixup",
            },
        )

        self.assertEqual(config["method"], "saliencymix")
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["method_prob"], 0.5)
        self.assertEqual(config["saliency_source"], "batch")
        self.assertFalse(config["use_basic_augmentation"])
        self.assertEqual(config["sal_aug_recipe"], "tiny_openmixup")

    def test_saliencymix_alias_section_resolves(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "saliency_mix",
                "saliency_mix": {
                    "alpha": 1.1,
                    "prob": 0.4,
                    "no_repeat": True,
                    "saliency_source": "batch",
                },
                "sal_aug_recipe": "tiny_openmixup",
            },
        )

        self.assertEqual(config["method"], "saliencymix")
        self.assertEqual(config["alpha"], 1.1)
        self.assertEqual(config["method_prob"], 0.4)
        self.assertTrue(config["saliencymix_no_repeat"])
        self.assertEqual(config["saliency_source"], "batch")

    def test_saliencymix_alias_flat_fields_resolve(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "saliency-mix",
                "saliency_mix_alpha": 1.2,
                "saliency_mix_prob": 0.3,
                "saliency_mix_no_repeat": True,
                "saliency_mix_saliency_source": "batch",
                "sal_aug_recipe": "tiny_openmixup",
            },
        )

        self.assertEqual(config["method"], "saliencymix")
        self.assertEqual(config["alpha"], 1.2)
        self.assertEqual(config["method_prob"], 0.3)
        self.assertTrue(config["saliencymix_no_repeat"])
        self.assertEqual(config["saliency_source"], "batch")

    def test_no_augment_overrides_explicit_aug_recipe(self):
        config = resolved_config(
            _args(no_augment=True),
            {
                "dataset": "tiny_imagenet",
                "method": "baseline",
                "basic_aug": False,
                "aug_recipe": "tiny_openmixup",
            },
        )

        self.assertFalse(config["use_basic_augmentation"])
        self.assertEqual(config["aug_recipe"], "none")

    def test_no_augment_overrides_explicit_saliency_aug_recipe(self):
        config = resolved_config(
            _args(no_augment=True),
            {
                "dataset": "tiny_imagenet",
                "method": "saliencymix",
                "saliency_source": "batch",
                "sal_basic_aug": False,
                "sal_aug_recipe": "tiny_openmixup",
            },
        )

        self.assertFalse(config["sal_basic_aug"])
        self.assertEqual(config["sal_aug_recipe"], "none")

    def test_string_boolean_config_values_are_parsed_strictly(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "mixup",
                "basic_aug": "false",
                "cross_device_shuffle": "true",
                "mixup_no_repeat": "false",
                "save_checkpoint": "false",
                "save_best_only": "true",
                "final_test": "true",
                "eval_on_test_each_epoch": "false",
                "save_csv": "true",
            },
        )

        self.assertFalse(config["use_basic_augmentation"])
        self.assertTrue(config["cross_device_shuffle"])
        self.assertFalse(config["mixup_no_repeat"])
        self.assertFalse(config["save_checkpoint"])
        self.assertTrue(config["save_best_only"])
        self.assertTrue(config["final_test"])
        self.assertFalse(config["eval_on_test_each_epoch"])
        self.assertTrue(config["save_csv"])

    def test_invalid_string_boolean_config_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "basic_aug must be a boolean"):
            resolved_config(
                _args(),
                {
                    "dataset": "tiny_imagenet",
                    "method": "baseline",
                    "basic_aug": "definitely",
                },
            )

    def test_tiny_saliencymix_xla4_uses_cached_batch_saliency_source(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["method"], "saliencymix")
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["method_prob"], 0.5)
        self.assertEqual(config["saliency_source"], "batch")
        self.assertEqual(config["saliency_dir"], "./data")
        self.assertFalse(config["validate_saliency_cache_on_load"])

    def test_resolved_config_can_enable_full_saliency_cache_scan_on_load(self):
        config = resolved_config(
            _args(),
            {
                "dataset": "tiny_imagenet",
                "method": "saliencymix",
                "saliency_source": "batch",
                "basic_aug": False,
                "validate_saliency_cache_on_load": True,
            },
        )

        self.assertTrue(config["validate_saliency_cache_on_load"])

    def test_non_batch_saliencymix_smoke_reuses_paired_saliency_aug_recipe(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
        config = resolved_config(_args(saliency_source="gradient"), raw_config)

        self.assertEqual(config["saliency_source"], "gradient")
        self.assertFalse(config["use_basic_augmentation"])
        self.assertEqual(config["aug_recipe"], "tiny_openmixup")
        self.assertEqual(config["sal_aug_recipe"], "tiny_openmixup")
        self.assertFalse(training_needs_batch_saliency_maps(config, _args()))

    def test_batch_saliency_rejects_unpaired_basic_augmentation(self):
        with self.assertRaisesRegex(ValueError, "cached saliency maps stay aligned"):
            resolved_config(
                _args(),
                {
                    "dataset": "tiny_imagenet",
                    "method": "saliencymix",
                    "saliency_source": "batch",
                    "basic_aug": True,
                    "sal_aug_recipe": "tiny_openmixup",
                },
            )

    def test_tiny_guided_sr_xla4_uses_online_spectral_residual_saliency(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
        config = resolved_config(_args(), raw_config)

        self.assertEqual(config["method"], "guided_sr")
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["alpha"], 1.0)
        self.assertEqual(config["method_prob"], 0.5)
        self.assertEqual(config["guidedmixup_condition"], "greedy")
        self.assertEqual(config["saliency_source"], "spectral_residual")
        self.assertIsNone(config["saliency_path"])
        self.assertEqual(config["aug_recipe"], "tiny_openmixup")
        self.assertEqual(config["sal_aug_recipe"], "none")
        self.assertFalse(config["cross_device_shuffle"])

    def test_saliency_dir_override_does_not_create_guided_sr_cache_path(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
        config = resolved_config(_args(saliency_dir="/mnt/cache"), raw_config)

        self.assertEqual(config["saliency_dir"], "/mnt/cache")
        self.assertIsNone(config["saliency_path"])

    def test_guided_sr_batch_source_defaults_to_guided_cache_path(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
        config = resolved_config(_args(saliency_source="batch"), raw_config)

        self.assertEqual(config["saliency_source"], "batch")
        self.assertEqual(config["saliency_dir"], "./data")
        self.assertEqual(config["saliency_path"], "data/tiny_imagenet_train_guided_sr_saliency.npy")
        self.assertEqual(config["aug_recipe"], "none")
        self.assertEqual(config["sal_aug_recipe"], "tiny_openmixup")

    def test_guided_sr_batch_source_cache_path_follows_saliency_dir_override(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
        config = resolved_config(_args(saliency_source="batch", saliency_dir="/mnt/cache"), raw_config)

        self.assertEqual(config["saliency_dir"], "/mnt/cache")
        self.assertEqual(config["saliency_path"], "/mnt/cache/tiny_imagenet_train_guided_sr_saliency.npy")
        self.assertEqual(config["aug_recipe"], "none")
        self.assertEqual(config["sal_aug_recipe"], "tiny_openmixup")

    def test_saliency_dir_override_relocates_relative_cli_saliency_path(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
        config = resolved_config(_args(saliency_dir="/mnt/cache", saliency_path="maps.npy"), raw_config)

        self.assertEqual(config["saliency_dir"], "/mnt/cache")
        self.assertEqual(config["saliency_path"], "/mnt/cache/maps.npy")

    def test_data_dir_override_moves_default_saliencymix_cache_dir(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
        config = resolved_config(_args(data_dir="/mnt/tiny"), raw_config)

        self.assertEqual(config["data_dir"], "/mnt/tiny")
        self.assertEqual(config["saliency_dir"], "/mnt/tiny")
        self.assertIsNone(config["saliency_path"])

    def test_data_dir_override_does_not_create_guided_sr_cache_path(self):
        raw_config = load_config("configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml")
        config = resolved_config(_args(data_dir="/mnt/tiny"), raw_config)

        self.assertEqual(config["data_dir"], "/mnt/tiny")
        self.assertEqual(config["saliency_dir"], "/mnt/tiny")
        self.assertIsNone(config["saliency_path"])

    def test_data_dir_override_keeps_independent_saliency_dir(self):
        config = resolved_config(
            _args(data_dir="/mnt/tiny"),
            {
                "dataset": "tiny_imagenet",
                "data_dir": "./data",
                "method": "saliencymix",
                "saliency_source": "batch",
                "saliency_dir": "./cache",
                "basic_aug": False,
            },
        )

        self.assertEqual(config["data_dir"], "/mnt/tiny")
        self.assertEqual(config["saliency_dir"], "./cache")

    def test_validate_global_batch_size_enforces_xla_world_size(self):
        config = {"batch_size": 32, "global_batch_size": 128}

        validate_global_batch_size(config, world_size=4, use_xla=True)

        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            validate_global_batch_size(config, world_size=1, use_xla=True)

    def test_validate_global_batch_size_ignores_cpu_debug_runs(self):
        config = {"batch_size": 32, "global_batch_size": 128}

        validate_global_batch_size(config, world_size=1, use_xla=False)

    def test_xla_method_probability_decision_is_rank_independent(self):
        config = {"method_prob": 0.5}
        args = _args(seed=123)

        decision_a = should_apply_method(config, args, epoch=3, step=17, use_xla=True, world_size=4)
        decision_b = should_apply_method(config, args, epoch=3, step=17, use_xla=True, world_size=4)

        self.assertEqual(decision_a, decision_b)

    def test_xla_method_probability_decision_ignores_global_random_state(self):
        config = {"method_prob": 0.5}
        args = _args(seed=123)

        random.seed(1)
        for _ in range(11):
            random.random()
        decision_a = should_apply_method(config, args, epoch=3, step=17, use_xla=True, world_size=4)

        random.seed(999)
        for _ in range(37):
            random.random()
        decision_b = should_apply_method(config, args, epoch=3, step=17, use_xla=True, world_size=4)

        self.assertEqual(decision_a, decision_b)

    def test_method_probability_boundaries_do_not_sample_random(self):
        args = _args(seed=123)

        with patch("allthemix.cli.train.random.random", side_effect=AssertionError("random should not be sampled")):
            self.assertFalse(should_apply_method({"method_prob": 0.0}, args, epoch=1, step=1, use_xla=False, world_size=1))
            self.assertTrue(should_apply_method({"method_prob": 1.0}, args, epoch=1, step=1, use_xla=False, world_size=1))
            self.assertFalse(should_apply_method({"method_prob": 0.0}, args, epoch=1, step=1, use_xla=True, world_size=4))
            self.assertTrue(should_apply_method({"method_prob": 1.0}, args, epoch=1, step=1, use_xla=True, world_size=4))

    def test_xla_method_probability_uses_seed_epoch_step_schedule(self):
        config = {"method_prob": 0.5}
        args = _args(seed=123)
        expected = random.Random((123 + 1) * 1_000_003 + 3 * 10_007 + 17).random() < 0.5

        decision = should_apply_method(config, args, epoch=3, step=17, use_xla=True, world_size=4)

        self.assertEqual(decision, expected)


if __name__ == "__main__":
    unittest.main()
