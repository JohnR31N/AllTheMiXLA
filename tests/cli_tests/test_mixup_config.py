import argparse
import unittest

from allthemix.cli.train import load_config, resolved_config


def _args(**overrides):
    defaults = {
        "dataset": None,
        "recipe": None,
        "method": None,
        "data_dir": None,
        "output_dir": None,
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
        "checkpoint": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class MixUpConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
