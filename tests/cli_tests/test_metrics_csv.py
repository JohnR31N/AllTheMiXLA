import csv
import tempfile
import unittest
from pathlib import Path

from allthemix.cli.train import append_metrics_csv, metrics_csv_path, pct_to_error, pct_to_fraction


class MetricsCsvTests(unittest.TestCase):
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
                    "val_top1": "60.09",
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
                },
            )

            lines = path.read_text().strip().splitlines()

        self.assertEqual(lines[0].split(",")[:3], ["epoch", "phase", "lr"])
        self.assertIn("train_accuracy", lines[0])
        self.assertIn("eval_top1_error", lines[0])
        self.assertIn("test_top1_error", lines[0])
        self.assertIn("train_val", lines[1])
        self.assertIn("0.399100", lines[1])
        self.assertIn("final_test", lines[2])

    def test_append_metrics_csv_migrates_old_percent_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            path.write_text("epoch,phase,train_top1,val_top1,best_top1,test_top1\n1,train_val,50.0,60.09,60.09,\n")

            append_metrics_csv(path, {"epoch": 2, "phase": "train_val", "val_top1": "66.57", "best_top1": "66.57"})

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["train_accuracy"], "0.500000")
        self.assertEqual(rows[0]["eval_top1_error"], "0.399100")
        self.assertEqual(rows[0]["best_top1_error"], "0.399100")
        self.assertEqual(rows[1]["eval_top1_accuracy"], "0.665700")
        self.assertEqual(rows[1]["eval_top1_error"], "0.334300")

    def test_accuracy_helpers_use_fraction_scale(self):
        self.assertEqual(pct_to_fraction(66.57), 0.6657)
        self.assertAlmostEqual(pct_to_error(66.57), 0.3343)


if __name__ == "__main__":
    unittest.main()
