import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from allthemix.cli.train import (
    RUN_METADATA_COMPATIBILITY_KEYS,
    archive_checkpoint_artifacts_for_fresh_run,
    checkpoint_backup_path,
    checkpoint_metadata_for_validation,
    checkpoint_metadata_path,
    clone_model_state_dict,
    eval_only_best_checkpoint_to_load,
    load_checkpoint_metadata_for_validation,
    load_model_checkpoint,
    restore_best_weights_for_final_test,
    restore_required_best_weights_for_final_test,
    restore_training_state,
    save_checkpoint,
    temporary_checkpoint_metadata_path,
    temporary_checkpoint_path,
    validate_checkpoint_metadata_matches_config,
)
from allthemix.networks.classifiers import ImageClassifier
from allthemix.networks.heads import LinearHead


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _train_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(4, 2)).sum()
    loss.backward()
    optimizer.step()


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
            "batch_size": 32,
            "global_batch_size": 128,
            "method_prob": 1.0,
            "run_metadata_required": True,
        }
    )
    config.update(overrides)
    return config


class _TinyFlatKeyBackbone(torch.nn.Module):
    output_dim = 2

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.conv1(x)


class CheckpointResumeTests(unittest.TestCase):
    def test_archive_checkpoint_artifacts_for_fresh_run_moves_resume_files_and_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir)
            checkpoint_paths = [
                checkpoint_root / "best.pt",
                checkpoint_root / "last.pt",
                checkpoint_root / "epoch_0005.pt",
            ]
            for path in checkpoint_paths:
                path.write_text(path.name)
                checkpoint_metadata_path(path).write_text(f"{path.name}.json")
            untouched = checkpoint_root / "manual.pt"
            untouched.write_text("manual")

            archived = archive_checkpoint_artifacts_for_fresh_run(checkpoint_root)

            remaining_auto_files = [path for path in checkpoint_paths if path.exists()]
            remaining_sidecars = [checkpoint_metadata_path(path) for path in checkpoint_paths if checkpoint_metadata_path(path).exists()]
            archived_names = [path.name for path in archived]
            untouched_exists = untouched.exists()

        self.assertEqual(remaining_auto_files, [])
        self.assertEqual(remaining_sidecars, [])
        self.assertEqual(len(archived), 6)
        self.assertTrue(all(".stale-" in name for name in archived_names))
        self.assertTrue(untouched_exists)

    def test_save_checkpoint_records_best_epoch(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch=5,
                best_acc=61.25,
                config={"dataset": "tinyimagenet"},
                use_xla=False,
                xm=None,
                best_epoch=3,
            )

            checkpoint = _load_checkpoint(path)
            metadata_path = path.with_suffix(path.suffix + ".json")
            metadata_text = metadata_path.read_text()

        self.assertEqual(checkpoint["epoch"], 5)
        self.assertEqual(checkpoint["best_epoch"], 3)
        self.assertEqual(checkpoint["best_acc"], 61.25)
        self.assertIn("optimizer", checkpoint)
        self.assertIn("scheduler", checkpoint)
        self.assertIn('"epoch": 5', metadata_text)
        self.assertIn('"best_epoch": 3', metadata_text)

    def test_save_checkpoint_removes_temporary_files_after_atomic_write(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch=1,
                best_acc=50.0,
                config={"dataset": "tinyimagenet"},
                use_xla=False,
                xm=None,
            )
            temp_checkpoint_exists = temporary_checkpoint_path(path).exists()
            temp_metadata_exists = temporary_checkpoint_metadata_path(path).exists()

        self.assertFalse(temp_checkpoint_exists)
        self.assertFalse(temp_metadata_exists)

    def test_save_checkpoint_keeps_existing_files_when_atomic_metadata_write_fails(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch=1,
                best_acc=50.0,
                config={"dataset": "old"},
                use_xla=False,
                xm=None,
            )
            old_checkpoint = _load_checkpoint(path)
            old_metadata = json.loads(checkpoint_metadata_path(path).read_text())
            original_write_text = Path.write_text

            def fail_temp_metadata_write(write_path, *args, **kwargs):
                if Path(write_path).name == temporary_checkpoint_metadata_path(path).name:
                    raise OSError("simulated checkpoint metadata write failure")
                return original_write_text(write_path, *args, **kwargs)

            with (
                patch.object(Path, "write_text", fail_temp_metadata_write),
                self.assertRaisesRegex(OSError, "simulated checkpoint metadata write failure"),
            ):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=2,
                    best_acc=60.0,
                    config={"dataset": "new"},
                    use_xla=False,
                    xm=None,
                )

            checkpoint = _load_checkpoint(path)
            metadata = json.loads(checkpoint_metadata_path(path).read_text())
            temp_checkpoint_exists = temporary_checkpoint_path(path).exists()
            temp_metadata_exists = temporary_checkpoint_metadata_path(path).exists()

        self.assertEqual(checkpoint["epoch"], old_checkpoint["epoch"])
        self.assertEqual(checkpoint["config"], old_checkpoint["config"])
        self.assertEqual(metadata, old_metadata)
        self.assertFalse(temp_checkpoint_exists)
        self.assertFalse(temp_metadata_exists)

    def test_save_checkpoint_rolls_back_existing_files_when_atomic_metadata_replace_fails(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch=1,
                best_acc=50.0,
                config={"dataset": "old"},
                use_xla=False,
                xm=None,
            )
            old_checkpoint = _load_checkpoint(path)
            old_metadata = json.loads(checkpoint_metadata_path(path).read_text())
            original_replace = Path.replace

            def fail_temp_metadata_replace(replace_path, target, *args, **kwargs):
                if Path(replace_path).name == temporary_checkpoint_metadata_path(path).name:
                    raise OSError("simulated checkpoint metadata replace failure")
                return original_replace(replace_path, target, *args, **kwargs)

            with (
                patch.object(Path, "replace", fail_temp_metadata_replace),
                self.assertRaisesRegex(OSError, "simulated checkpoint metadata replace failure"),
            ):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=2,
                    best_acc=60.0,
                    config={"dataset": "new"},
                    use_xla=False,
                    xm=None,
                )

            checkpoint = _load_checkpoint(path)
            metadata = json.loads(checkpoint_metadata_path(path).read_text())
            leftover_paths = [
                temporary_checkpoint_path(path),
                temporary_checkpoint_metadata_path(path),
                checkpoint_backup_path(path),
                checkpoint_backup_path(checkpoint_metadata_path(path)),
            ]

        self.assertEqual(checkpoint["epoch"], old_checkpoint["epoch"])
        self.assertEqual(checkpoint["config"], old_checkpoint["config"])
        self.assertEqual(metadata, old_metadata)
        for leftover_path in leftover_paths:
            self.assertFalse(leftover_path.exists())

    def test_save_checkpoint_removes_new_checkpoint_when_atomic_metadata_replace_fails_without_existing_files(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            original_replace = Path.replace

            def fail_temp_metadata_replace(replace_path, target, *args, **kwargs):
                if Path(replace_path).name == temporary_checkpoint_metadata_path(path).name:
                    raise OSError("simulated checkpoint metadata replace failure without old checkpoint")
                return original_replace(replace_path, target, *args, **kwargs)

            with (
                patch.object(Path, "replace", fail_temp_metadata_replace),
                self.assertRaisesRegex(OSError, "without old checkpoint"),
            ):
                save_checkpoint(
                    path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=1,
                    best_acc=50.0,
                    config={"dataset": "tinyimagenet"},
                    use_xla=False,
                    xm=None,
                )

            leftover_paths = [
                path,
                checkpoint_metadata_path(path),
                temporary_checkpoint_path(path),
                temporary_checkpoint_metadata_path(path),
                checkpoint_backup_path(path),
                checkpoint_backup_path(checkpoint_metadata_path(path)),
            ]

        for leftover_path in leftover_paths:
            self.assertFalse(leftover_path.exists())

    def test_load_model_checkpoint_requests_full_checkpoint_payload(self):
        model = torch.nn.Linear(2, 1)
        payload = {"model": model.state_dict(), "epoch": 3, "config": {"dataset": "tinyimagenet"}}
        calls = []

        def fake_torch_load(path, **kwargs):
            calls.append((path, kwargs))
            return payload

        with patch("allthemix.cli.train.torch.load", fake_torch_load):
            metadata = load_model_checkpoint("best.pt", model)

        self.assertEqual(metadata["epoch"], 3)
        self.assertEqual(calls, [("best.pt", {"map_location": "cpu", "weights_only": False})])

    def test_load_model_checkpoint_supports_old_torch_load_without_weights_only(self):
        model = torch.nn.Linear(2, 1)
        payload = {"model": model.state_dict(), "epoch": 4}
        calls = []

        def fake_torch_load(path, **kwargs):
            calls.append((path, kwargs))
            if "weights_only" in kwargs:
                raise TypeError("weights_only is not supported")
            return payload

        with patch("allthemix.cli.train.torch.load", fake_torch_load):
            metadata = load_model_checkpoint("legacy.pt", model)

        self.assertEqual(metadata["epoch"], 4)
        self.assertEqual(
            calls,
            [
                ("legacy.pt", {"map_location": "cpu", "weights_only": False}),
                ("legacy.pt", {"map_location": "cpu"}),
            ],
        )

    def test_load_model_checkpoint_maps_flat_backbone_and_fc_keys(self):
        model = ImageClassifier(_TinyFlatKeyBackbone(), LinearHead(in_features=2, num_classes=3))
        flat_state = {
            "conv1.weight": torch.full_like(model.backbone.conv1.weight, 2.0),
            "fc.weight": torch.full_like(model.head.fc.weight, 3.0),
            "fc.bias": torch.full_like(model.head.fc.bias, 4.0),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "official_flat.pt"
            torch.save({"state_dict": flat_state, "epoch": 9}, path)

            metadata = load_model_checkpoint(path, model)

        self.assertEqual(metadata["epoch"], 9)
        torch.testing.assert_close(model.backbone.conv1.weight, flat_state["conv1.weight"])
        torch.testing.assert_close(model.head.fc.weight, flat_state["fc.weight"])
        torch.testing.assert_close(model.head.fc.bias, flat_state["fc.bias"])

    def test_checkpoint_metadata_validation_rejects_missing_model_impl_version(self):
        config = _compatible_checkpoint_config()
        old_config = dict(config)
        old_config.pop("model_impl_version")

        with self.assertRaisesRegex(RuntimeError, "model_impl_version"):
            validate_checkpoint_metadata_matches_config("old.pt", {"config": old_config}, config)

    def test_checkpoint_metadata_validation_accepts_canonical_model_alias(self):
        config = _compatible_checkpoint_config(model="preact_resnet18")
        alias_config = dict(config)
        alias_config["model"] = "preact-resnet18"

        validate_checkpoint_metadata_matches_config("alias.pt", {"config": alias_config}, config)

    def test_checkpoint_metadata_validation_accepts_dataset_and_method_aliases(self):
        config = _compatible_checkpoint_config(dataset="tinyimagenet", method="guided_sr")
        alias_config = dict(config)
        alias_config["dataset"] = "tiny_imagenet"
        alias_config["method"] = "guidedmixup"

        validate_checkpoint_metadata_matches_config("alias.pt", {"config": alias_config}, config)

    def test_checkpoint_metadata_validation_parses_boolean_strings_strictly(self):
        config = _compatible_checkpoint_config(run_metadata_required=True)
        string_true_config = dict(config)
        string_true_config["run_metadata_required"] = "true"
        string_false_config = dict(config)
        string_false_config["run_metadata_required"] = "false"

        validate_checkpoint_metadata_matches_config("string_true.pt", {"config": string_true_config}, config)
        with self.assertRaisesRegex(RuntimeError, "run_metadata_required"):
            validate_checkpoint_metadata_matches_config("string_false.pt", {"config": string_false_config}, config)

    def test_checkpoint_metadata_validation_rejects_different_augmentation_protocol(self):
        config = _compatible_checkpoint_config(
            use_basic_augmentation=False,
            aug_recipe="none",
            transform_profile="openmixup",
        )
        checkpoint_config = dict(config)
        checkpoint_config["use_basic_augmentation"] = True
        checkpoint_config["aug_recipe"] = "tiny_openmixup"

        with self.assertRaisesRegex(RuntimeError, "use_basic_augmentation"):
            validate_checkpoint_metadata_matches_config("aug.pt", {"config": checkpoint_config}, config)

        checkpoint_config = dict(config)
        checkpoint_config["transform_profile"] = "paper"

        with self.assertRaisesRegex(RuntimeError, "transform_profile"):
            validate_checkpoint_metadata_matches_config("aug.pt", {"config": checkpoint_config}, config)

    def test_checkpoint_metadata_validation_allows_weight_only_external_checkpoint_by_default(self):
        config = _compatible_checkpoint_config()

        validate_checkpoint_metadata_matches_config("external.pt", {"epoch": 1}, config)

        with self.assertRaisesRegex(RuntimeError, "no config metadata"):
            validate_checkpoint_metadata_matches_config(
                "external.pt",
                {"epoch": 1},
                config,
                require_config=True,
            )

    def test_load_checkpoint_metadata_for_validation_prefers_json_sidecar(self):
        config = _compatible_checkpoint_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best.pt"
            torch.save({"epoch": 1, "config": {"method": "old"}}, path)
            path.with_suffix(path.suffix + ".json").write_text(
                '{"epoch": 2, "config": {"method": "baseline", "model_impl_version": 2}}'
            )

            metadata = load_checkpoint_metadata_for_validation(path)

        self.assertEqual(metadata["epoch"], 2)
        self.assertEqual(metadata["config"]["method"], "baseline")

    def test_checkpoint_metadata_validation_prefers_incompatible_sidecar_over_payload(self):
        config = _compatible_checkpoint_config()
        old_config = dict(config)
        old_config.pop("model_impl_version")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "last.pt"
            torch.save({"epoch": 1, "model": torch.nn.Linear(2, 1).state_dict()}, path)
            path.with_suffix(path.suffix + ".json").write_text(
                json.dumps({"epoch": 1, "config": old_config})
            )

            metadata = checkpoint_metadata_for_validation(path, {"epoch": 1})

        with self.assertRaisesRegex(RuntimeError, "model_impl_version"):
            validate_checkpoint_metadata_matches_config(path, metadata, config)

    def test_restore_best_weights_for_final_test_prefers_memory_snapshot(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.5)
        best_state = clone_model_state_dict(model)
        with torch.no_grad():
            model.weight.fill_(4.0)
            model.bias.fill_(2.0)

        source = restore_best_weights_for_final_test(
            model,
            best_state,
            Path("missing_best.pt"),
            torch.device("cpu"),
        )

        self.assertEqual(source, "memory")
        self.assertTrue(torch.equal(model.weight, torch.ones_like(model.weight)))
        self.assertTrue(torch.equal(model.bias, torch.full_like(model.bias, 0.5)))

    def test_restore_best_weights_for_final_test_falls_back_to_best_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch=7,
                best_acc=70.0,
                config={"dataset": "tinyimagenet"},
                use_xla=False,
                xm=None,
                best_epoch=7,
            )
            with torch.no_grad():
                model.weight.fill_(0.0)
                model.bias.fill_(0.0)

            source = restore_best_weights_for_final_test(model, None, path, torch.device("cpu"))

        self.assertEqual(source, path.as_posix())
        self.assertTrue(torch.equal(model.weight, torch.full_like(model.weight, 3.0)))
        self.assertTrue(torch.equal(model.bias, torch.full_like(model.bias, 1.0)))

    def test_restore_required_best_weights_for_final_test_rejects_missing_best(self):
        model = torch.nn.Linear(2, 1)

        with self.assertRaisesRegex(RuntimeError, "final_test_checkpoint=best"):
            restore_required_best_weights_for_final_test(
                model,
                None,
                Path("missing_best.pt"),
                torch.device("cpu"),
            )

    def test_eval_only_best_checkpoint_loader_prefers_default_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir) / "checkpoints"
            checkpoint_root.mkdir()
            best_path = checkpoint_root / "best.pt"
            last_path = checkpoint_root / "last.pt"
            best_path.write_bytes(b"best")
            last_path.write_bytes(b"last")

            self.assertEqual(eval_only_best_checkpoint_to_load(None, checkpoint_root), best_path)
            self.assertEqual(eval_only_best_checkpoint_to_load(last_path, checkpoint_root), best_path)
            self.assertIsNone(eval_only_best_checkpoint_to_load(best_path, checkpoint_root))

    def test_eval_only_best_checkpoint_loader_keeps_explicit_best_like_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir) / "checkpoints"
            checkpoint_root.mkdir()
            default_best_path = checkpoint_root / "best.pt"
            explicit_best_path = Path(tmpdir) / "external" / "saliencymix_best.pt"
            default_best_path.write_bytes(b"default-best")
            explicit_best_path.parent.mkdir()
            explicit_best_path.write_bytes(b"explicit-best")

            self.assertIsNone(
                eval_only_best_checkpoint_to_load(
                    explicit_best_path,
                    checkpoint_root,
                    explicit_checkpoint=True,
                )
            )

    def test_eval_only_best_checkpoint_loader_rejects_explicit_last_without_default_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir) / "checkpoints"
            checkpoint_root.mkdir()
            last_path = checkpoint_root / "last.pt"
            last_path.write_bytes(b"last")

            with self.assertRaisesRegex(RuntimeError, "explicit last checkpoint"):
                eval_only_best_checkpoint_to_load(last_path, checkpoint_root, explicit_checkpoint=True)

    def test_eval_only_best_checkpoint_loader_replaces_explicit_last_with_default_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir) / "checkpoints"
            checkpoint_root.mkdir()
            best_path = checkpoint_root / "best.pt"
            last_path = checkpoint_root / "last.pt"
            best_path.write_bytes(b"best")
            last_path.write_bytes(b"last")

            self.assertEqual(
                eval_only_best_checkpoint_to_load(last_path, checkpoint_root, explicit_checkpoint=True),
                best_path,
            )

    def test_eval_only_best_checkpoint_loader_rejects_last_when_best_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_root = Path(tmpdir) / "checkpoints"
            checkpoint_root.mkdir()
            last_path = checkpoint_root / "last.pt"
            last_path.write_bytes(b"last")

            with self.assertRaisesRegex(RuntimeError, "requires a best checkpoint"):
                eval_only_best_checkpoint_to_load(last_path, checkpoint_root)

            self.assertIsNone(eval_only_best_checkpoint_to_load(checkpoint_root / "best.pt", checkpoint_root))

    def test_restore_training_state_resumes_epoch_optimizer_and_scheduler(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
        _train_step(model, optimizer)
        scheduler.step()
        _train_step(model, optimizer)
        scheduler.step()

        checkpoint = {
            "epoch": 2,
            "best_acc": 66.5,
            "best_epoch": 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        restored_model = torch.nn.Linear(2, 1)
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
        restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            restored_optimizer,
            milestones=[1],
            gamma=0.1,
        )

        start_epoch, best_acc, best_epoch, resumed = restore_training_state(
            checkpoint,
            restored_optimizer,
            restored_scheduler,
            torch.device("cpu"),
        )

        self.assertTrue(resumed)
        self.assertEqual(start_epoch, 3)
        self.assertEqual(best_acc, 66.5)
        self.assertEqual(best_epoch, 1)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
        for state in restored_optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device.type, "cpu")

    def test_restore_training_state_ignores_weight_only_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

        start_epoch, best_acc, best_epoch, resumed = restore_training_state(
            {"model": model.state_dict(), "epoch": 4},
            optimizer,
            scheduler,
            torch.device("cpu"),
        )

        self.assertFalse(resumed)
        self.assertEqual(start_epoch, 1)
        self.assertEqual(best_acc, 0.0)
        self.assertEqual(best_epoch, 0)

    def test_restore_training_state_rejects_optimizer_without_scheduler(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
        _train_step(model, optimizer)
        checkpoint = {
            "epoch": 20,
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
        }
        restored_model = torch.nn.Linear(2, 1)
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
        restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            restored_optimizer,
            milestones=[1],
            gamma=0.1,
        )

        with self.assertRaisesRegex(RuntimeError, "incomplete training state"):
            restore_training_state(
                checkpoint,
                restored_optimizer,
                restored_scheduler,
                torch.device("cpu"),
            )

    def test_restore_training_state_rejects_scheduler_without_optimizer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
        _train_step(model, optimizer)
        scheduler.step()
        checkpoint = {
            "epoch": 20,
            "optimizer": None,
            "scheduler": scheduler.state_dict(),
        }
        restored_model = torch.nn.Linear(2, 1)
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)
        restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            restored_optimizer,
            milestones=[1],
            gamma=0.1,
        )

        with self.assertRaisesRegex(RuntimeError, "incomplete training state"):
            restore_training_state(
                checkpoint,
                restored_optimizer,
                restored_scheduler,
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
