import tempfile
import unittest
from pathlib import Path

from allthemix.cli.train import append_metrics_csv, metrics_csv_path


class MetricsCsvTests(unittest.TestCase):
    def test_metrics_csv_path_defaults_to_metrics_csv(self):
        path = metrics_csv_path(Path("runs/example"), {"output_name": ""})

        self.assertEqual(path, Path("runs/example/metrics.csv"))

    def test_append_metrics_csv_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.csv"
            append_metrics_csv(path, {"epoch": 1, "phase": "train_val", "val_top1": "60.09"})
            append_metrics_csv(path, {"epoch": 200, "phase": "final_test", "test_top1": "59.44"})

            lines = path.read_text().strip().splitlines()

        self.assertEqual(lines[0].split(",")[:3], ["epoch", "phase", "lr"])
        self.assertIn("train_val", lines[1])
        self.assertIn("final_test", lines[2])


if __name__ == "__main__":
    unittest.main()
