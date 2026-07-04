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
        self.assertEqual(config["output_dir"], "./runs/mixup")

    def test_cli_method_overrides_config(self):
        raw_config = load_config("configs/cifar10/preact_resnet18/fmix.yaml")
        config = resolved_config(_args(method="mixup", alpha=0.4, mix_prob=0.5), raw_config)

        self.assertEqual(config["method"], "mixup")
        self.assertEqual(config["alpha"], 0.4)
        self.assertEqual(config["method_prob"], 0.5)


if __name__ == "__main__":
    unittest.main()
