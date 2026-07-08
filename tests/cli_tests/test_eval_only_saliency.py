import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from allthemix.cli.train import RUN_METADATA_COMPATIBILITY_KEYS
from allthemix.cli.train import parse_args, run_worker


def _compatible_checkpoint_config(**overrides):
    config = {key: None for key in RUN_METADATA_COMPATIBILITY_KEYS}
    config.update(
        {
            "dataset": "tinyimagenet",
            "recipe": "openmixup",
            "model": "preact_resnet18",
            "model_impl_version": 2,
            "method": "baseline",
            "epochs": 200,
            "batch_size": 2,
            "method_prob": 1.0,
            "run_metadata_required": True,
        }
    )
    config.update(overrides)
    return config


class EvalOnlySaliencyTests(unittest.TestCase):
    def test_eval_only_best_checkpoint_uses_default_best_without_random_weight_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "eval_only_default_best.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "recipe: openmixup",
                        "model: preact_resnet18",
                        "method: baseline",
                        "batch_size: 2",
                        "epochs: 200",
                        "validation_split: 0.5",
                        "final_test: false",
                        "final_test_checkpoint: best",
                        "save_csv: false",
                        "save_checkpoint: false",
                        f"output_dir: {root.as_posix()}/outputs",
                        "run_name: eval_only_default_best",
                        f"checkpoint_dir: {root.as_posix()}/checkpoints",
                    ]
                )
            )
            best_path = root / "checkpoints" / "eval_only_default_best" / "best.pt"
            best_path.parent.mkdir(parents=True)
            best_path.write_bytes(b"best")
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--eval-only",
                    "--num-workers",
                    "0",
                    "--log-interval",
                    "0",
                ]
            )
            train_set = TensorDataset(torch.randn(6, 3, 4, 4), torch.arange(6) % 2)
            eval_train_set = TensorDataset(torch.randn(6, 3, 4, 4), torch.arange(6) % 2)
            test_set = TensorDataset(torch.randn(4, 3, 4, 4), torch.arange(4) % 2)
            dataset_calls = []

            def fake_build_datasets(*build_args, **build_kwargs):
                dataset_calls.append((build_args, build_kwargs))
                if len(dataset_calls) == 1:
                    return train_set, test_set
                return eval_train_set, test_set

            model = torch.nn.Linear(1, 2)
            with (
                patch("allthemix.cli.train.build_datasets", side_effect=fake_build_datasets),
                patch("allthemix.cli.train.build_model", return_value=model),
                patch("allthemix.cli.train.load_model_checkpoint", return_value={"epoch": 200, "best_acc": 60.0}) as load_model_checkpoint,
                patch("allthemix.cli.train.evaluate", return_value=(1.0, 60.0, 80.0)),
                patch("allthemix.cli.train.print_master") as print_master,
            ):
                run_worker(0, args)

            load_model_checkpoint.assert_called_once_with(best_path, model)
            messages = [str(call.args[0]) for call in print_master.call_args_list]
            self.assertTrue(any("Loaded best checkpoint" in message for message in messages))
            self.assertFalse(any("randomly initialized" in message for message in messages))
            self.assertFalse(any("evaluating current weights" in message for message in messages))

    def test_eval_only_best_checkpoint_keeps_explicit_external_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "eval_only_external_best.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "recipe: openmixup",
                        "model: preact_resnet18",
                        "method: baseline",
                        "batch_size: 2",
                        "epochs: 200",
                        "validation_split: 0.5",
                        "final_test: false",
                        "final_test_checkpoint: best",
                        "save_csv: false",
                        "save_checkpoint: false",
                        f"output_dir: {root.as_posix()}/outputs",
                        "run_name: eval_only_external_best",
                        f"checkpoint_dir: {root.as_posix()}/checkpoints",
                    ]
                )
            )
            default_best_path = root / "checkpoints" / "eval_only_external_best" / "best.pt"
            explicit_best_path = root / "external" / "saliencymix_best.pt"
            default_best_path.parent.mkdir(parents=True)
            explicit_best_path.parent.mkdir()
            default_best_path.write_bytes(b"default-best")
            explicit_best_path.write_bytes(b"explicit-best")
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--eval-only",
                    "--checkpoint",
                    str(explicit_best_path),
                    "--num-workers",
                    "0",
                    "--log-interval",
                    "0",
                ]
            )
            train_set = TensorDataset(torch.randn(6, 3, 4, 4), torch.arange(6) % 2)
            test_set = TensorDataset(torch.randn(4, 3, 4, 4), torch.arange(4) % 2)
            model = torch.nn.Linear(1, 2)

            with (
                patch("allthemix.cli.train.build_datasets", return_value=(train_set, test_set)),
                patch("allthemix.cli.train.build_model", return_value=model),
                patch("allthemix.cli.train.load_model_checkpoint", return_value={"epoch": 200, "best_acc": 60.0}) as load_model_checkpoint,
                patch("allthemix.cli.train.evaluate", return_value=(1.0, 60.0, 80.0)),
                patch("allthemix.cli.train.print_master"),
            ):
                run_worker(0, args)

            load_model_checkpoint.assert_called_once_with(str(explicit_best_path), model)

    def test_eval_only_explicit_checkpoint_rejects_incompatible_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "eval_only_sidecar.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "recipe: openmixup",
                        "model: preact_resnet18",
                        "method: baseline",
                        "batch_size: 2",
                        "epochs: 200",
                        "validation_split: 0.0",
                        "final_test: false",
                        "save_csv: false",
                        "save_checkpoint: false",
                        "run_metadata_required: true",
                        f"output_dir: {root.as_posix()}/outputs",
                        "run_name: eval_only_sidecar",
                    ]
                )
            )
            model = torch.nn.Linear(1, 2)
            checkpoint_path = root / "external" / "best.pt"
            checkpoint_path.parent.mkdir()
            torch.save({"epoch": 200, "model": model.state_dict()}, checkpoint_path)
            old_config = _compatible_checkpoint_config()
            old_config.pop("model_impl_version")
            checkpoint_path.with_suffix(checkpoint_path.suffix + ".json").write_text(
                json.dumps({"epoch": 200, "config": old_config})
            )
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--eval-only",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--num-workers",
                    "0",
                    "--log-interval",
                    "0",
                ]
            )
            train_set = TensorDataset(torch.randn(4, 3, 4, 4), torch.arange(4) % 2)
            val_set = TensorDataset(torch.randn(4, 3, 4, 4), torch.arange(4) % 2)

            with (
                patch("allthemix.cli.train.build_datasets", return_value=(train_set, val_set)),
                patch("allthemix.cli.train.build_model", return_value=model),
                patch("allthemix.cli.train.evaluate", return_value=(1.0, 60.0, 80.0)),
            ):
                with self.assertRaisesRegex(RuntimeError, "model_impl_version"):
                    run_worker(0, args)

    def test_eval_only_batch_saliency_skips_train_cache_attach(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "saliency_eval_only.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "recipe: openmixup",
                        "model: preact_resnet18",
                        "method: saliencymix",
                        "saliency_source: batch",
                        "saliency_dir: ./missing-cache",
                        "batch_size: 2",
                        "epochs: 200",
                        "validation_split: 0.5",
                        "final_test: false",
                        "final_test_checkpoint: best",
                        "save_csv: false",
                        "save_checkpoint: false",
                        f"output_dir: {root.as_posix()}/outputs",
                        "run_name: eval_only_saliency",
                        f"checkpoint_dir: {root.as_posix()}/checkpoints",
                    ]
                )
            )
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--eval-only",
                    "--checkpoint",
                    str(root / "best.pt"),
                    "--num-workers",
                    "0",
                    "--log-interval",
                    "0",
                ]
            )
            train_set = TensorDataset(torch.randn(6, 3, 4, 4), torch.arange(6) % 2)
            eval_train_set = TensorDataset(torch.randn(6, 3, 4, 4), torch.arange(6) % 2)
            test_set = TensorDataset(torch.randn(4, 3, 4, 4), torch.arange(4) % 2)
            dataset_calls = []

            def fake_build_datasets(*build_args, **build_kwargs):
                dataset_calls.append((build_args, build_kwargs))
                if len(dataset_calls) == 1:
                    return train_set, test_set
                return eval_train_set, test_set

            with (
                patch("allthemix.cli.train.build_datasets", side_effect=fake_build_datasets),
                patch("allthemix.cli.train.attach_train_saliency_maps") as attach_train_saliency_maps,
                patch("allthemix.cli.train.build_model", return_value=torch.nn.Linear(1, 2)),
                patch("allthemix.cli.train.load_model_checkpoint", return_value={"epoch": 200, "best_acc": 60.0}),
                patch("allthemix.cli.train.evaluate", return_value=(1.0, 60.0, 80.0)),
                patch("allthemix.cli.train.print_master"),
            ):
                run_worker(0, args)

            attach_train_saliency_maps.assert_not_called()
            self.assertTrue(dataset_calls[0][1]["normalize_train"])


if __name__ == "__main__":
    unittest.main()
