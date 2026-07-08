import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from allthemix.cli.summarize import TINY_IMAGENET_XLA4_SPECS
from allthemix.cli.train import (
    build_batch_mixer,
    load_config,
    make_scheduler,
    parse_args,
    resolved_config,
    run_worker,
    save_checkpoint,
    train_one_epoch,
)
from allthemix.data.saliency_dataset import attach_train_saliency_maps as real_attach_train_saliency_maps


class _TinyHookModel(torch.nn.Module):
    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.fc = torch.nn.Linear(4, int(num_classes))

    def forward(self, images, feature_hook=None):
        features = self.conv(images)
        if feature_hook is not None:
            features = feature_hook(features, 1)
        features = F.relu(features).mean(dim=(-2, -1))
        return self.fc(features)


def _write_tiny_imagenet_original_layout(root: Path) -> Path:
    tiny_root = root / "data" / "tiny-imagenet-200"
    classes = [f"n{index:08d}" for index in range(1, 5)]
    colors = [(220, 30, 30), (30, 180, 60), (40, 90, 220), (210, 180, 40)]
    (tiny_root / "val" / "images").mkdir(parents=True)
    (tiny_root / "wnids.txt").write_text("\n".join(classes) + "\n")

    val_rows = []
    for class_index, class_name in enumerate(classes):
        train_dir = tiny_root / "train" / class_name / "images"
        train_dir.mkdir(parents=True)
        for item in range(4):
            image = Image.new("RGB", (72, 72), colors[class_index])
            image.putpixel((item + 2, item + 2), (255, 255, 255))
            image.save(train_dir / f"{class_name}_{item}.JPEG", format="JPEG")

        for item in range(2):
            image_name = f"val_{class_index}_{item}.JPEG"
            image = Image.new("RGB", (72, 72), colors[class_index])
            image.putpixel((item + 4, item + 4), (0, 0, 0))
            image.save(tiny_root / "val" / "images" / image_name, format="JPEG")
            val_rows.append(f"{image_name}\t{class_name}\t0\t0\t64\t64")

    (tiny_root / "val" / "val_annotations.txt").write_text("\n".join(val_rows) + "\n")
    return tiny_root


class TrainLoopSmokeTests(unittest.TestCase):
    def test_all_tiny_xla4_configs_reach_run_worker_one_cpu_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = torch.rand(12, 3, 64, 64)
            labels = torch.arange(12, dtype=torch.long) % 200
            train_set = TensorDataset(images, labels)
            val_set = TensorDataset(images[:4], labels[:4])

            for spec in TINY_IMAGENET_XLA4_SPECS:
                with self.subTest(method=spec.method_key):
                    raw_config = load_config(spec.config_path)
                    raw_config.update(
                        {
                            "data_dir": str(root / "data"),
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "checkpoints"),
                            "save_checkpoint": False,
                        }
                    )
                    config_path = root / f"{spec.method_key}.json"
                    config_path.write_text(json.dumps(raw_config))
                    cli_args = [
                        "--config",
                        str(config_path),
                        "--device",
                        "cpu",
                        "--num-workers",
                        "0",
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--max-train-steps",
                        "1",
                        "--max-val-steps",
                        "1",
                        "--log-interval",
                        "0",
                    ]
                    if spec.method_key == "saliencymix":
                        cli_args.extend(["--saliency-source", "gradient"])
                    args = parse_args(cli_args)

                    with (
                        patch("allthemix.cli.train.build_datasets", return_value=(train_set, val_set)),
                        patch("allthemix.cli.train.build_model", return_value=_TinyHookModel()),
                        patch("allthemix.cli.train.print_master"),
                    ):
                        run_worker(0, args)

                    metrics_path = root / "outputs" / raw_config["run_name"] / "metrics.csv"
                    with metrics_path.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))

                    self.assertEqual(rows[0]["phase"], "train_val")
                    self.assertNotEqual(rows[0]["train_top1_error"], "")

    def test_tiny_xla4_configs_run_against_real_tiny_imagenet_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_imagenet_original_layout(root)

            for spec in TINY_IMAGENET_XLA4_SPECS:
                with self.subTest(method=spec.method_key):
                    raw_config = load_config(spec.config_path)
                    raw_config.update(
                        {
                            "data_dir": str(root / "data"),
                            "output_dir": str(root / "outputs"),
                            "checkpoint_dir": str(root / "checkpoints"),
                            "run_name": f"{spec.method_key}_real_tiny_smoke",
                            "save_checkpoint": False,
                        }
                    )
                    config_path = root / f"{spec.method_key}.json"
                    config_path.write_text(json.dumps(raw_config))
                    cli_args = [
                        "--config",
                        str(config_path),
                        "--device",
                        "cpu",
                        "--num-workers",
                        "0",
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--max-train-steps",
                        "1",
                        "--max-val-steps",
                        "1",
                        "--log-interval",
                        "0",
                    ]
                    if spec.method_key == "saliencymix":
                        cli_args.extend(["--saliency-source", "gradient"])
                    args = parse_args(cli_args)

                    with (
                        patch("allthemix.cli.train.build_model", return_value=_TinyHookModel()),
                        patch("allthemix.cli.train.print_master"),
                    ):
                        run_worker(0, args)

                    metrics_path = root / "outputs" / raw_config["run_name"] / "metrics.csv"
                    with metrics_path.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    metadata = json.loads((metrics_path.parent / "config.json").read_text())

                    self.assertEqual([row["phase"] for row in rows], ["train_val", "final_test"])
                    self.assertNotEqual(rows[0]["train_top1_error"], "")
                    self.assertNotEqual(rows[-1]["test_top1_error"], "")
                    self.assertNotEqual(rows[-1]["best_top1_error"], "")
                    if spec.method_key == "guided_sr":
                        self.assertEqual(metadata["resolved"]["saliency_source"], "spectral_residual")

    def test_saliencymix_xla4_runs_against_real_tiny_layout_with_cached_batch_saliency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_imagenet_original_layout(root)
            data_dir = root / "data"
            np.save(data_dir / "tiny_imagenet_train_saliency.npy", np.ones((16, 72, 72), dtype=np.float32))

            raw_config = load_config("configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
            raw_config.update(
                {
                    "output_dir": str(root / "outputs"),
                    "checkpoint_dir": str(root / "checkpoints"),
                    "run_name": "saliencymix_real_tiny_batch_cache_smoke",
                    "save_checkpoint": False,
                }
            )
            config_path = root / "saliencymix_batch_cache.json"
            config_path.write_text(json.dumps(raw_config))
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--data-dir",
                    str(data_dir),
                    "--device",
                    "cpu",
                    "--num-workers",
                    "0",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--max-train-steps",
                    "1",
                    "--max-val-steps",
                    "1",
                    "--mix-prob",
                    "1.0",
                    "--log-interval",
                    "0",
                ]
            )

            with (
                patch("allthemix.cli.train.build_model", return_value=_TinyHookModel()),
                patch("allthemix.cli.train.print_master"),
            ):
                run_worker(0, args)

            metrics_path = root / "outputs" / raw_config["run_name"] / "metrics.csv"
            with metrics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            metadata = json.loads((metrics_path.parent / "config.json").read_text())

        self.assertEqual([row["phase"] for row in rows], ["train_val", "final_test"])
        self.assertNotEqual(rows[0]["train_top1_error"], "")
        self.assertNotEqual(rows[-1]["test_top1_error"], "")
        self.assertEqual(metadata["resolved"]["saliency_source"], "batch")
        self.assertEqual(metadata["resolved"]["sal_aug_recipe"], "tiny_openmixup")

    def test_xla_distributed_train_sampler_sets_epoch_each_epoch(self):
        class RecordingDistributedSampler:
            instances = []

            def __init__(self, dataset, num_replicas, rank, shuffle):
                self.dataset = dataset
                self.num_replicas = int(num_replicas)
                self.rank = int(rank)
                self.shuffle = bool(shuffle)
                self.epochs = []
                RecordingDistributedSampler.instances.append(self)

            def set_epoch(self, epoch):
                self.epochs.append(int(epoch))

            def __iter__(self):
                return iter(range(self.rank, len(self.dataset), self.num_replicas))

            def __len__(self):
                dataset_len = len(self.dataset)
                if dataset_len <= self.rank:
                    return 0
                return ((dataset_len - 1 - self.rank) // self.num_replicas) + 1

        class FakeXm:
            def xla_device(self):
                return torch.device("cpu")

        class FakePl:
            @staticmethod
            def MpDeviceLoader(loader, device):
                del device
                return loader

        class FakeXr:
            @staticmethod
            def global_ordinal():
                return 0

            @staticmethod
            def world_size():
                return 2

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_set = TensorDataset(
                torch.rand(8, 3, 8, 8),
                torch.arange(8, dtype=torch.long) % 200,
            )
            val_set = TensorDataset(
                torch.rand(4, 3, 8, 8),
                torch.arange(4, dtype=torch.long) % 200,
            )
            config = {
                "dataset": "tiny_imagenet",
                "data_dir": str(root / "data"),
                "model": "preact_resnet18",
                "method": "baseline",
                "batch_size": 2,
                "global_batch_size": 4,
                "epochs": 2,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "lr_schedule": "step",
                "lr_decay_epochs": [],
                "validation_split": 0.0,
                "final_test": False,
                "basic_aug": False,
                "save_csv": False,
                "output_dir": str(root / "outputs"),
                "run_name": "xla_sampler_epoch_smoke",
                "save_checkpoint": False,
                "checkpoint_dir": str(root / "checkpoints"),
                "seed": 0,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "xla",
                    "--num-workers",
                    "0",
                    "--log-interval",
                    "0",
                ]
            )
            train_epochs = []

            def fake_train_one_epoch(
                model,
                loader,
                optimizer,
                mixer,
                device,
                epoch,
                config,
                args,
                use_xla,
                xm,
                rank,
                world_size,
            ):
                del model, mixer, device, config, args, use_xla, xm
                optimizer.zero_grad(set_to_none=True)
                optimizer.step()
                train_epochs.append((int(epoch), int(rank), int(world_size), len(loader)))
                return 1.0, 10.0

            with (
                patch(
                    "allthemix.cli.train._optional_xla_import",
                    return_value={"xm": FakeXm(), "pl": FakePl, "xr": FakeXr()},
                ),
                patch("allthemix.cli.train.DistributedSampler", RecordingDistributedSampler),
                patch("allthemix.cli.train.build_datasets", return_value=(train_set, val_set)),
                patch("allthemix.cli.train.build_model", return_value=_TinyHookModel()),
                patch("allthemix.cli.train.train_one_epoch", side_effect=fake_train_one_epoch),
                patch("allthemix.cli.train.evaluate", return_value=(1.0, 20.0, 30.0)),
                patch("allthemix.cli.train.print_master"),
            ):
                run_worker(0, args)

        self.assertEqual(len(RecordingDistributedSampler.instances), 1)
        sampler = RecordingDistributedSampler.instances[0]
        self.assertEqual(sampler.epochs, [1, 2])
        self.assertTrue(sampler.shuffle)
        self.assertEqual(train_epochs, [(1, 0, 2, 2), (2, 0, 2, 2)])

    def test_all_table_methods_complete_one_train_step(self):
        base_config = {
            "method_prob": 1.0,
            "alpha": 1.0,
            "reformulate": False,
            "decay_power": 3.0,
            "max_soft": 0.0,
            "image_size": 8,
            "cross_device_shuffle": False,
            "mixup_no_repeat": False,
            "fmix_no_repeat": False,
            "cutmix_no_repeat": False,
            "resizemix_scope_min": 0.1,
            "resizemix_scope_max": 0.8,
            "resizemix_use_alpha": False,
            "resizemix_no_repeat": False,
            "saliencymix_no_repeat": False,
            "guidedmixup_blur_kernel": 3,
            "guidedmixup_condition": "random",
            "saliency_source": "gradient",
            "catchupmix_cutmix_alpha": 1.0,
            "catchupmix_num_layers": 1,
            "catchupmix_no_repeat": False,
        }
        methods = ["mixup", "cutmix", "resizemix", "fmix", "saliencymix", "guided_sr", "catchupmix"]

        for method in methods:
            with self.subTest(method=method):
                torch.manual_seed(0)
                images = torch.randn(4, 3, 8, 8)
                targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                if method == "saliencymix":
                    saliency_maps = torch.ones(4, 1, 8, 8)
                    loader = DataLoader(TensorDataset(images, targets, saliency_maps), batch_size=4)
                    config = {**base_config, "method": method, "saliency_source": "batch"}
                else:
                    loader = DataLoader(TensorDataset(images, targets), batch_size=4)
                    config = {**base_config, "method": method}
                model = _TinyHookModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                mixer = build_batch_mixer(config)

                train_loss, train_acc = train_one_epoch(
                    model,
                    loader,
                    optimizer,
                    mixer,
                    torch.device("cpu"),
                    epoch=1,
                    config=config,
                    args=SimpleNamespace(max_train_steps=1, log_interval=0, seed=0),
                    use_xla=False,
                    xm=None,
                )

                self.assertTrue(torch.isfinite(torch.tensor(train_loss)))
                self.assertGreaterEqual(train_acc, 0.0)
                self.assertLessEqual(train_acc, 100.0)

    def test_train_loop_passes_catchupmix_feature_hook_to_model(self):
        class RecordingHookModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
                self.fc = torch.nn.Linear(4, 4)
                self.saw_feature_hook = False

            def forward(self, images, feature_hook=None):
                features = self.conv(images)
                if feature_hook is not None:
                    self.saw_feature_hook = True
                    features = feature_hook(features, 1)
                return self.fc(F.relu(features).mean(dim=(-2, -1)))

        images = torch.randn(4, 3, 8, 8)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        config = {
            "method": "catchupmix",
            "method_prob": 1.0,
            "alpha": 1.0,
            "cross_device_shuffle": False,
            "catchupmix_cutmix_alpha": 1.0,
            "catchupmix_num_layers": 1,
            "catchupmix_no_repeat": False,
        }
        model = RecordingHookModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with patch("allthemix.methods.catchupmix.random.randint", return_value=1):
            train_loss, train_acc = train_one_epoch(
                model,
                loader,
                optimizer,
                build_batch_mixer(config),
                torch.device("cpu"),
                epoch=1,
                config=config,
                args=SimpleNamespace(max_train_steps=1, log_interval=0, seed=0),
                use_xla=False,
                xm=None,
            )

        self.assertTrue(model.saw_feature_hook)
        self.assertTrue(torch.isfinite(torch.tensor(train_loss)))
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 100.0)

    def test_split_protocol_writes_best_checkpoint_final_test_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
            val_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
            train_set = TensorDataset(torch.randn(6, 3, 2, 2), train_labels)
            eval_train_set = TensorDataset(torch.randn(6, 3, 2, 2), train_labels)
            val_set = TensorDataset(torch.randn(4, 3, 2, 2), val_labels)
            config = {
                "dataset": "tiny_imagenet",
                "data_dir": str(root / "data"),
                "model": "preact_resnet18",
                "method": "baseline",
                "batch_size": 2,
                "epochs": 1,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "lr_schedule": "step",
                "lr_decay_epochs": [],
                "validation_split": 0.5,
                "eval_on_test_each_epoch": False,
                "final_test": True,
                "final_test_checkpoint": "best",
                "basic_aug": False,
                "aug_recipe": "none",
                "save_csv": True,
                "output_dir": str(root / "outputs"),
                "run_name": "split_final_test_smoke",
                "save_checkpoint": False,
                "checkpoint_dir": str(root / "checkpoints"),
                "seed": 0,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--num-workers",
                    "0",
                    "--max-train-steps",
                    "1",
                    "--max-val-steps",
                    "1",
                    "--log-interval",
                    "0",
                ]
            )
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 200))

            with (
                patch(
                    "allthemix.cli.train.build_datasets",
                    side_effect=[(train_set, val_set), (eval_train_set, val_set)],
                ),
                patch("allthemix.cli.train.build_model", return_value=model),
                patch("allthemix.cli.train.print_master"),
            ):
                run_worker(0, args)

            metrics_path = root / "outputs" / "split_final_test_smoke" / "metrics.csv"
            with metrics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([row["phase"] for row in rows], ["train_val", "final_test"])
        self.assertEqual(rows[-1]["epoch"], "1")
        self.assertNotEqual(rows[-1]["test_top1_error"], "")
        self.assertNotEqual(rows[-1]["best_top1_error"], "")
        self.assertEqual(rows[-1]["final_test_checkpoint"], "best")
        self.assertEqual(rows[-1]["final_test_checkpoint_source"], "memory")

    def test_resume_final_test_best_checkpoint_uses_historical_best_when_not_refreshed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_name = "resume_best_final_test_smoke"
            raw_config = {
                "dataset": "tiny_imagenet",
                "data_dir": str(root / "data"),
                "model": "preact_resnet18",
                "method": "baseline",
                "batch_size": 2,
                "epochs": 2,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "lr_schedule": "step",
                "lr_decay_epochs": [],
                "validation_split": 0.5,
                "eval_on_test_each_epoch": False,
                "final_test": True,
                "final_test_checkpoint": "best",
                "basic_aug": False,
                "aug_recipe": "none",
                "save_csv": False,
                "output_dir": str(root / "outputs"),
                "run_name": run_name,
                "save_checkpoint": False,
                "checkpoint_dir": str(root / "checkpoints"),
                "run_metadata_required": True,
                "seed": 0,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(raw_config))
            checkpoint_args = parse_args(["--config", str(config_path), "--device", "cpu", "--num-workers", "0"])
            checkpoint_config = resolved_config(checkpoint_args, raw_config)
            checkpoint_root = root / "checkpoints" / run_name

            checkpoint_model = _TinyHookModel()
            optimizer = torch.optim.SGD(
                checkpoint_model.parameters(),
                lr=float(checkpoint_config["lr"]),
                momentum=float(checkpoint_config["momentum"]),
                weight_decay=float(checkpoint_config["weight_decay"]),
            )
            scheduler = make_scheduler(optimizer, checkpoint_config)

            with torch.no_grad():
                for parameter in checkpoint_model.parameters():
                    parameter.fill_(7.0)
            save_checkpoint(
                checkpoint_root / "best.pt",
                checkpoint_model,
                optimizer,
                scheduler,
                epoch=1,
                best_acc=90.0,
                config=checkpoint_config,
                use_xla=False,
                xm=None,
                best_epoch=1,
            )

            with torch.no_grad():
                for parameter in checkpoint_model.parameters():
                    parameter.fill_(1.0)
            save_checkpoint(
                checkpoint_root / "last.pt",
                checkpoint_model,
                optimizer,
                scheduler,
                epoch=1,
                best_acc=90.0,
                config=checkpoint_config,
                use_xla=False,
                xm=None,
                best_epoch=1,
            )

            train_labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
            val_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
            train_set = TensorDataset(torch.randn(6, 3, 2, 2), train_labels)
            eval_train_set = TensorDataset(torch.randn(6, 3, 2, 2), train_labels)
            val_set = TensorDataset(torch.randn(4, 3, 2, 2), val_labels)
            final_seen = {}

            def fake_train_one_epoch(model, loader, optimizer, mixer, device, epoch, config, args, use_xla, xm, *rest):
                del model, loader, mixer, device, epoch, config, args, use_xla, xm, rest
                optimizer.zero_grad(set_to_none=True)
                optimizer.step()
                return 1.0, 10.0

            def fake_final_test(
                model,
                test_loader,
                device,
                epoch,
                config,
                args,
                use_xla,
                xm,
                xr,
                csv_path,
                best_acc,
                best_epoch,
                checkpoint_source=None,
            ):
                del test_loader, device, epoch, config, args, use_xla, xm, xr, csv_path
                final_seen["weight"] = float(next(model.parameters()).flatten()[0].detach().cpu())
                final_seen["best_acc"] = best_acc
                final_seen["best_epoch"] = best_epoch
                return 0.0, 0.0, 0.0

            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--num-workers",
                    "0",
                    "--checkpoint",
                    str(checkpoint_root / "last.pt"),
                    "--max-train-steps",
                    "1",
                    "--max-val-steps",
                    "1",
                    "--log-interval",
                    "0",
                ]
            )

            with (
                patch(
                    "allthemix.cli.train.build_datasets",
                    side_effect=[(train_set, val_set), (eval_train_set, val_set)],
                ),
                patch("allthemix.cli.train.build_model", return_value=_TinyHookModel()),
                patch("allthemix.cli.train.train_one_epoch", side_effect=fake_train_one_epoch),
                patch("allthemix.cli.train.evaluate", return_value=(1.0, 80.0, 95.0)),
                patch("allthemix.cli.train.evaluate_final_test", side_effect=fake_final_test),
                patch("allthemix.cli.train.print_master"),
            ):
                run_worker(0, args)

        self.assertEqual(final_seen["weight"], 7.0)
        self.assertEqual(final_seen["best_acc"], 90.0)
        self.assertEqual(final_seen["best_epoch"], 1)

    def test_run_worker_trains_with_cached_batch_saliency_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            cache_path = data_dir / "tiny_imagenet_train_saliency.npy"
            data_dir.mkdir()
            np.save(cache_path, np.ones((4, 8, 8), dtype=np.float32))
            train_set = TensorDataset(torch.rand(4, 3, 8, 8), torch.tensor([0, 1, 0, 1], dtype=torch.long))
            val_set = TensorDataset(torch.rand(2, 3, 8, 8), torch.tensor([0, 1], dtype=torch.long))
            config = {
                "dataset": "tiny_imagenet",
                "data_dir": str(data_dir),
                "model": "preact_resnet18",
                "method": "saliencymix",
                "batch_size": 2,
                "epochs": 1,
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "lr_schedule": "step",
                "lr_decay_epochs": [],
                "validation_split": 0.0,
                "final_test": False,
                "basic_aug": False,
                "sal_basic_aug": False,
                "sal_aug_recipe": "none",
                "saliency_source": "batch",
                "saliency_dir": str(data_dir),
                "save_csv": True,
                "output_dir": str(root / "outputs"),
                "run_name": "batch_saliency_smoke",
                "save_checkpoint": False,
                "seed": 0,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--num-workers",
                    "0",
                    "--max-train-steps",
                    "1",
                    "--max-val-steps",
                    "1",
                    "--log-interval",
                    "0",
                ]
            )

            attach_kwargs = {}

            def capture_attach(*attach_args, **kwargs):
                attach_kwargs.update(kwargs)
                return real_attach_train_saliency_maps(*attach_args, **kwargs)

            with (
                patch("allthemix.cli.train.build_datasets", return_value=(train_set, val_set)) as build_datasets_mock,
                patch("allthemix.cli.train.attach_train_saliency_maps", side_effect=capture_attach),
                patch("allthemix.cli.train.build_model", return_value=_TinyHookModel()),
                patch("allthemix.cli.train.print_master"),
            ):
                run_worker(0, args)

            self.assertFalse(build_datasets_mock.call_args.kwargs["normalize_train"])
            self.assertFalse(attach_kwargs["validate_finite"])
            self.assertTrue(attach_kwargs["validate_sample_finite"])
            metrics_path = root / "outputs" / "batch_saliency_smoke" / "metrics.csv"
            with metrics_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["phase"], "train_val")
        self.assertNotEqual(rows[0]["train_top1_error"], "")


if __name__ == "__main__":
    unittest.main()
