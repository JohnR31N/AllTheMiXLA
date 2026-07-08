import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from allthemix.cli.summarize import RESUME_COMPATIBILITY_KEYS
from allthemix.cli.train import (
    RUN_METADATA_COMPATIBILITY_KEYS,
    SequentialDistributedEvalSampler,
    append_eval_metrics_csv,
    append_final_test_metrics_csv,
    append_metrics_csv,
    metrics_csv_path,
    pct_to_error,
    pct_to_fraction,
    prepare_metrics_csv_for_run,
    temporary_metrics_csv_path,
    temporary_run_metadata_path,
    evaluate,
    topk_correct_tensor,
    write_run_metadata_atomic,
)


def _compatible_run_config(**overrides):
    config = {key: None for key in RUN_METADATA_COMPATIBILITY_KEYS}
    config.update(
        {
            "run_metadata_required": True,
            "batch_size": 32,
            "global_batch_size": 128,
            "model": "preact_resnet18",
            "model_impl_version": 2,
            "method": "baseline",
            "method_prob": 1.0,
        }
    )
    config.update(overrides)
    return config


class MetricsCsvTests(unittest.TestCase):
    def test_run_metadata_keys_match_summary_compatibility_keys(self):
        self.assertEqual(RUN_METADATA_COMPATIBILITY_KEYS, RESUME_COMPATIBILITY_KEYS)

    def test_metrics_csv_path_defaults_to_metrics_csv(self):
        path = metrics_csv_path(Path("runs/example"), {"output_name": ""})

        self.assertEqual(path, Path("runs/example/metrics.csv"))

    def test_append_metrics_csv_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            append_metrics_csv(
                path,
                {
                    "epoch": 1,
                    "phase": "train_val",
                    "train_accuracy": "0.500000",
                    "eval_top1_error": "0.399100",
                    "eval_top5_accuracy": "0.900000",
                    "val_top1": "60.09",
                    "val_top5": "90.00",
                },
            )
            append_metrics_csv(
                path,
                {
                    "epoch": 200,
                    "phase": "final_test",
                    "test_top1_accuracy": "0.594400",
                    "test_top1_error": "0.405600",
                    "test_top1": "59.44",
                    "test_top5_accuracy": "0.880000",
                    "test_top5_error": "0.120000",
                    "test_top5": "88.00",
                },
            )

            lines = path.read_text().strip().splitlines()

        self.assertEqual(lines[0].split(",")[:3], ["epoch", "phase", "lr"])
        self.assertIn("train_accuracy", lines[0])
        self.assertIn("eval_top1_error", lines[0])
        self.assertIn("eval_top5_accuracy", lines[0])
        self.assertIn("eval_top5_error", lines[0])
        self.assertIn("test_top1_error", lines[0])
        self.assertIn("test_top5_error", lines[0])
        self.assertIn("train_val", lines[1])
        self.assertIn("0.399100", lines[1])
        self.assertIn("final_test", lines[2])

    def test_append_metrics_csv_removes_temporary_file_after_atomic_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"

            append_metrics_csv(path, {"epoch": 1, "phase": "train_val", "val_top1": "60.09"})

            temp_exists = temporary_metrics_csv_path(path).exists()

        self.assertFalse(temp_exists)

    def test_append_metrics_csv_keeps_existing_file_when_atomic_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            append_metrics_csv(path, {"epoch": 1, "phase": "train_val", "val_top1": "60.09"})
            old_text = path.read_text()
            original_replace = Path.replace

            def fail_temp_metrics_replace(replace_path, target, *args, **kwargs):
                if Path(replace_path).name == temporary_metrics_csv_path(path).name:
                    raise OSError("simulated metrics replace failure")
                return original_replace(replace_path, target, *args, **kwargs)

            with (
                patch.object(Path, "replace", fail_temp_metrics_replace),
                self.assertRaisesRegex(OSError, "simulated metrics replace failure"),
            ):
                append_metrics_csv(path, {"epoch": 2, "phase": "train_val", "val_top1": "66.57"})

            new_text = path.read_text()
            temp_exists = temporary_metrics_csv_path(path).exists()

        self.assertEqual(new_text, old_text)
        self.assertFalse(temp_exists)

    def test_append_metrics_csv_removes_new_file_when_atomic_replace_fails_without_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            original_replace = Path.replace

            def fail_temp_metrics_replace(replace_path, target, *args, **kwargs):
                if Path(replace_path).name == temporary_metrics_csv_path(path).name:
                    raise OSError("simulated metrics replace failure without old file")
                return original_replace(replace_path, target, *args, **kwargs)

            with (
                patch.object(Path, "replace", fail_temp_metrics_replace),
                self.assertRaisesRegex(OSError, "without old file"),
            ):
                append_metrics_csv(path, {"epoch": 1, "phase": "train_val", "val_top1": "60.09"})

            path_exists = path.exists()
            temp_exists = temporary_metrics_csv_path(path).exists()

        self.assertFalse(path_exists)
        self.assertFalse(temp_exists)

    def test_eval_only_metric_helpers_write_table_epoch_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            append_eval_metrics_csv(
                path,
                epoch=200,
                val_loss=1.25,
                val_acc=66.5,
                val_top5=88.0,
                best_acc=67.0,
                best_epoch=177,
            )
            append_final_test_metrics_csv(
                path,
                epoch=200,
                test_loss=1.3,
                test_acc=65.25,
                test_top5=87.5,
                best_acc=67.0,
                best_epoch=177,
                final_test_checkpoint="best",
                final_test_checkpoint_source="memory",
            )

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["epoch"], "200")
        self.assertEqual(rows[0]["phase"], "eval")
        self.assertEqual(rows[0]["best_epoch"], "177")
        self.assertEqual(rows[0]["best_top1_error"], "0.330000")
        self.assertEqual(rows[1]["epoch"], "200")
        self.assertEqual(rows[1]["phase"], "final_test")
        self.assertEqual(rows[1]["test_top1_error"], "0.347500")
        self.assertEqual(rows[1]["test_top5_error"], "0.125000")
        self.assertEqual(rows[1]["final_test_checkpoint"], "best")
        self.assertEqual(rows[1]["final_test_checkpoint_source"], "memory")

    def test_append_metrics_csv_migrates_old_percent_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            path.write_text("epoch,phase,train_top1,val_top1,val_top5,best_top1,test_top1,test_top5\n1,train_val,50.0,60.09,90.0,60.09,,\n")

            append_metrics_csv(path, {"epoch": 2, "phase": "train_val", "val_top1": "66.57", "best_top1": "66.57"})

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["train_accuracy"], "0.500000")
        self.assertEqual(rows[0]["eval_top1_error"], "0.399100")
        self.assertEqual(rows[0]["eval_top5_accuracy"], "0.900000")
        self.assertEqual(rows[0]["eval_top5_error"], "0.100000")
        self.assertEqual(rows[0]["best_top1_error"], "0.399100")
        self.assertEqual(rows[1]["eval_top1_accuracy"], "0.665700")
        self.assertEqual(rows[1]["eval_top1_error"], "0.334300")

    def test_append_metrics_csv_migrates_percent_style_accuracy_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            path.write_text(
                "epoch,phase,eval_top1_accuracy,test_top1_accuracy,test_top5_accuracy\n"
                "1,final_test,66.57,59.44,88.00\n"
            )

            append_metrics_csv(path, {"epoch": 2, "phase": "train_val", "val_top1": "66.57"})

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["eval_top1_accuracy"], "0.665700")
        self.assertEqual(rows[0]["eval_top1_error"], "0.334300")
        self.assertEqual(rows[0]["test_top1_accuracy"], "0.594400")
        self.assertEqual(rows[0]["test_top1_error"], "0.405600")
        self.assertEqual(rows[0]["test_top5_accuracy"], "0.880000")
        self.assertEqual(rows[0]["test_top5_error"], "0.120000")

    def test_write_run_metadata_atomic_writes_config_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"

            write_run_metadata_atomic(config_path, {"resolved": {"dataset": "tinyimagenet"}})

            metadata = json.loads(config_path.read_text())
            temp_exists = temporary_run_metadata_path(config_path).exists()

        self.assertEqual(metadata["resolved"]["dataset"], "tinyimagenet")
        self.assertFalse(temp_exists)

    def test_write_run_metadata_atomic_keeps_existing_config_when_temp_write_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(json.dumps({"resolved": {"dataset": "old"}}))
            original_write_text = Path.write_text

            def fail_temp_config_write(write_path, *args, **kwargs):
                if Path(write_path).name == temporary_run_metadata_path(config_path).name:
                    raise OSError("simulated run metadata write failure")
                return original_write_text(write_path, *args, **kwargs)

            with (
                patch.object(Path, "write_text", fail_temp_config_write),
                self.assertRaisesRegex(OSError, "simulated run metadata write failure"),
            ):
                write_run_metadata_atomic(config_path, {"resolved": {"dataset": "new"}})

            metadata = json.loads(config_path.read_text())
            temp_exists = temporary_run_metadata_path(config_path).exists()

        self.assertEqual(metadata["resolved"]["dataset"], "old")
        self.assertFalse(temp_exists)

    def test_write_run_metadata_atomic_keeps_existing_config_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(json.dumps({"resolved": {"dataset": "old"}}))
            original_replace = Path.replace

            def fail_temp_config_replace(replace_path, target, *args, **kwargs):
                if Path(replace_path).name == temporary_run_metadata_path(config_path).name:
                    raise OSError("simulated run metadata replace failure")
                return original_replace(replace_path, target, *args, **kwargs)

            with (
                patch.object(Path, "replace", fail_temp_config_replace),
                self.assertRaisesRegex(OSError, "simulated run metadata replace failure"),
            ):
                write_run_metadata_atomic(config_path, {"resolved": {"dataset": "new"}})

            metadata = json.loads(config_path.read_text())
            temp_exists = temporary_run_metadata_path(config_path).exists()

        self.assertEqual(metadata["resolved"]["dataset"], "old")
        self.assertFalse(temp_exists)

    def test_prepare_metrics_csv_keeps_compatible_existing_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config()
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNone(archived)
            self.assertTrue(csv_path.exists())
            self.assertTrue(config_path.exists())

    def test_prepare_metrics_csv_archives_compatible_metrics_for_fresh_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config()
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config, archive_compatible=True)

            self.assertIsNotNone(archived)
            self.assertTrue(archived.exists())
            self.assertFalse(csv_path.exists())
            self.assertFalse(config_path.exists())

    def test_prepare_metrics_csv_keeps_metrics_when_only_saliency_cache_path_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config(
                method="saliencymix",
                saliency_source="batch",
                saliency_dir="/mnt/tiny",
                saliency_path="/mnt/tiny/tiny_imagenet_train_saliency.npy",
            )
            previous_config = dict(config)
            previous_config["saliency_dir"] = "./data"
            previous_config["saliency_path"] = "./data/tiny_imagenet_train_saliency.npy"
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": previous_config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNone(archived)
            self.assertTrue(csv_path.exists())
            self.assertTrue(config_path.exists())

    def test_prepare_metrics_csv_keeps_metrics_when_only_model_alias_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config(model="preact_resnet18")
            previous_config = dict(config)
            previous_config["model"] = "preact-resnet18"
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": previous_config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNone(archived)
            self.assertTrue(csv_path.exists())
            self.assertTrue(config_path.exists())

    def test_prepare_metrics_csv_keeps_metrics_when_only_dataset_or_method_alias_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config(dataset="tinyimagenet", method="guided_sr")
            previous_config = dict(config)
            previous_config["dataset"] = "tiny_imagenet"
            previous_config["method"] = "guidedmixup"
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": previous_config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNone(archived)
            self.assertTrue(csv_path.exists())
            self.assertTrue(config_path.exists())

    def test_prepare_metrics_csv_parses_boolean_metadata_strings_strictly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config(run_metadata_required=True, final_test=True)
            previous_config = dict(config)
            previous_config["run_metadata_required"] = "true"
            previous_config["final_test"] = "false"
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": previous_config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNotNone(archived)
            self.assertFalse(csv_path.exists())
            self.assertFalse(config_path.exists())

    def test_prepare_metrics_csv_archives_missing_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config()
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNotNone(archived)
            self.assertFalse(csv_path.exists())
            self.assertTrue(archived.exists())
            self.assertIn("stale", archived.name)

    def test_prepare_metrics_csv_archives_incompatible_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config(method="guided_sr", method_prob=1.0)
            old_config = dict(config)
            old_config["method_prob"] = 0.5
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": old_config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNotNone(archived)
            self.assertFalse(csv_path.exists())
            self.assertFalse(config_path.exists())
            self.assertTrue(archived.exists())
            self.assertTrue(list(run_dir.glob("config.stale-*.json")))

    def test_prepare_metrics_csv_archives_missing_model_impl_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config()
            old_config = dict(config)
            old_config.pop("model_impl_version")
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")
            config_path.write_text(json.dumps({"resolved": old_config}))

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNotNone(archived)
            self.assertFalse(csv_path.exists())
            self.assertFalse(config_path.exists())

    def test_prepare_metrics_csv_ignores_non_table_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            csv_path = run_dir / "metrics.csv"
            config_path = run_dir / "config.json"
            config = _compatible_run_config(run_metadata_required=False)
            csv_path.write_text("epoch,phase,test_top1_error\n200,final_test,0.40\n")

            archived = prepare_metrics_csv_for_run(csv_path, config_path, config)

            self.assertIsNone(archived)
            self.assertTrue(csv_path.exists())

    def test_accuracy_helpers_use_fraction_scale(self):
        self.assertEqual(pct_to_fraction(66.57), 0.6657)
        self.assertAlmostEqual(pct_to_error(66.57), 0.3343)

    def test_topk_correct_clamps_k_to_class_count(self):
        logits = torch.tensor([[0.1, 0.2, 0.9], [0.8, 0.1, 0.0]])
        targets = torch.tensor([0, 2])

        self.assertEqual(float(topk_correct_tensor(logits, targets, k=5)), 2.0)

    def test_eval_sampler_shards_without_padding_duplicates(self):
        shards = [list(SequentialDistributedEvalSampler(range(10), num_replicas=4, rank=rank)) for rank in range(4)]
        merged = [index for shard in shards for index in shard]

        self.assertEqual([len(shard) for shard in shards], [3, 3, 2, 2])
        self.assertEqual(sorted(merged), list(range(10)))
        self.assertEqual(len(set(merged)), 10)

    def test_eval_sampler_len_and_indices_cover_dataset_exactly(self):
        for dataset_size in range(15):
            for num_replicas in (1, 2, 3, 4, 8):
                with self.subTest(dataset_size=dataset_size, num_replicas=num_replicas):
                    samplers = [
                        SequentialDistributedEvalSampler(range(dataset_size), num_replicas=num_replicas, rank=rank)
                        for rank in range(num_replicas)
                    ]
                    shards = [list(sampler) for sampler in samplers]
                    merged = [index for shard in shards for index in shard]

                    self.assertEqual(sum(len(sampler) for sampler in samplers), dataset_size)
                    self.assertEqual(sorted(merged), list(range(dataset_size)))
                    self.assertEqual(len(set(merged)), dataset_size)

    def test_eval_sampler_allows_empty_tail_ranks(self):
        shards = [list(SequentialDistributedEvalSampler(range(2), num_replicas=4, rank=rank)) for rank in range(4)]

        self.assertEqual(shards, [[0], [1], [], []])

    def test_eval_sampler_rejects_invalid_replica_or_rank(self):
        with self.assertRaisesRegex(ValueError, "num_replicas"):
            SequentialDistributedEvalSampler(range(3), num_replicas=0, rank=0)

        with self.assertRaisesRegex(ValueError, "rank"):
            SequentialDistributedEvalSampler(range(3), num_replicas=2, rank=2)

    def test_evaluate_xla_reduces_top5_with_global_total(self):
        class FixedLogitModel(torch.nn.Module):
            def forward(self, images):
                return torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=images.device)

        class FakeXm:
            def __init__(self):
                self.reduced_names = []

            def mark_step(self):
                pass

            def mesh_reduce(self, name, value, reduce_fn):
                self.reduced_names.append(name)
                return reduce_fn([value, value])

        loader = DataLoader(
            TensorDataset(torch.zeros(2, 1), torch.tensor([0, 1])),
            batch_size=2,
        )
        xm = FakeXm()

        _, top1, top5 = evaluate(
            FixedLogitModel(),
            loader,
            torch.device("cpu"),
            epoch=3,
            config={"dataset": "tinyimagenet"},
            args=SimpleNamespace(max_val_steps=None),
            use_xla=True,
            xm=xm,
        )

        self.assertEqual(top1, 50.0)
        self.assertEqual(top5, 100.0)
        self.assertIn("val_3_correct_top5", xm.reduced_names)

    def test_evaluate_rejects_empty_global_sample_count(self):
        class UnusedModel(torch.nn.Module):
            def forward(self, images):
                return torch.empty((images.size(0), 2), device=images.device)

        loader = DataLoader(
            TensorDataset(torch.empty(0, 1), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )

        with self.assertRaisesRegex(RuntimeError, "processed 0 samples"):
            evaluate(
                UnusedModel(),
                loader,
                torch.device("cpu"),
                epoch=1,
                config={"dataset": "tinyimagenet"},
                args=SimpleNamespace(max_val_steps=None),
                use_xla=False,
                xm=None,
            )


if __name__ == "__main__":
    unittest.main()
