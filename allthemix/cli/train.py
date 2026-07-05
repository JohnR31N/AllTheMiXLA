"""Train MixUp/FMIX PreAct-ResNet18 on CIFAR-10/100 or Tiny-ImageNet.

Use ``python -m allthemix.cli.train --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from allthemix.data import attach_train_saliency_maps, build_datasets
from allthemix.data.datasets import IMAGENET_A_INDICES_IN_1K, IMAGENET_A_NUM_CLASSES
from allthemix.methods import FMix, GuidedSR, MixUp, SaliencyMix
from allthemix.networks import build_model
from allthemix.cli.presets import (
    DATASETS,
    RECIPES,
    get_dataset_preset,
    get_recipe_preset,
    normalize_dataset_name,
    preset_dict,
)
from allthemix.training.losses import fmix_cross_entropy, mixup_cross_entropy


METHOD_ALIASES = {
    "guided-sr": "guided_sr",
    "guidedsr": "guided_sr",
    "guidedmixup_sr": "guided_sr",
    "guidedmixup-sr": "guided_sr",
    "guided_mixup_sr": "guided_sr",
    "saliency_mix": "saliencymix",
    "saliency-mix": "saliencymix",
}
METHOD_CHOICES = sorted(
    {
        "baseline",
        "eval",
        "fmix",
        "guided_sr",
        "mixup",
        "none",
        "saliencymix",
        *METHOD_ALIASES.keys(),
    }
)


def normalize_method_name(name: Any) -> str:
    method = str(name).lower()
    return METHOD_ALIASES.get(method, method)


def _optional_xla_import() -> dict[str, Any] | None:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.runtime as xr
    except ModuleNotFoundError:
        return None
    return {"xm": xm, "pl": pl, "xr": xr}


def _optional_xla_launcher():
    try:
        import torch_xla
    except ModuleNotFoundError:
        return None
    if hasattr(torch_xla, "launch"):
        return torch_xla.launch

    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ModuleNotFoundError:
        return None

    def _spawn(fn, args=(), start_method="spawn", debug_single_process=False):
        if debug_single_process:
            return fn(0, *args)
        return xmp.spawn(fn, args=args, nprocs=None, start_method=start_method)

    return _spawn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MixUp/FMIX/SaliencyMix/Guided-SR PyTorch/XLA trainer")
    parser.add_argument("--config", default=None, help="YAML/JSON config path, e.g. configs/cifar10/preact_resnet18/fmix.yaml.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--recipe", choices=sorted(RECIPES), default=None)
    parser.add_argument("--method", choices=METHOD_CHOICES, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--download", action="store_true", default=None, help="Download CIFAR datasets if needed.")
    parser.add_argument("--no-augment", action="store_true", default=None, help="Disable train-time spatial augmentations.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=["cosine", "multistep", "step"], default=None)
    parser.add_argument("--milestones", type=int, nargs="*", default=None)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--decay-power", type=float, default=None)
    parser.add_argument("--max-soft", type=float, default=None)
    parser.add_argument("--reformulate", action="store_true", default=None)
    parser.add_argument("--fmix-prob", type=float, default=None)
    parser.add_argument("--mix-prob", type=float, default=None, help="Batch-level mix method probability.")
    parser.add_argument("--guidedmixup-blur-kernel", type=int, default=None)
    parser.add_argument("--guidedmixup-condition", choices=["random", "greedy"], default=None)
    parser.add_argument(
        "--saliency-source",
        choices=["batch", "gradient", "grad", "spectral_residual", "guided_sr", "sr", "online"],
        default=None,
    )
    parser.add_argument("--saliency-dir", default=None)
    parser.add_argument("--saliency-path", default=None)

    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xla"], default="auto")
    parser.add_argument("--num-cores", type=int, default=1, help="XLA processes to spawn when --device xla.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Limit steps per epoch for smoke tests.")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Limit validation steps for smoke tests.")
    parser.add_argument("--checkpoint", default=None, help="Load a model checkpoint before training/evaluation.")
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic epoch checkpoints; 0 disables.")
    return parser.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)
    text = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("YAML configs require PyYAML. Install it with `pip install PyYAML`.") from exc

    data = yaml.safe_load(text)
    return data or {}


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _first_config_value(raw_config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = raw_config.get(key)
        if value is not None:
            return value
    return default


def _choose(
    args: argparse.Namespace,
    raw_config: dict[str, Any],
    arg_name: str,
    section_name: str,
    key: str,
    default: Any,
) -> Any:
    cli_value = getattr(args, arg_name)
    if cli_value is not None:
        return cli_value
    section_value = _section(raw_config, section_name).get(key)
    if section_value is not None:
        return section_value
    return raw_config.get(key, default)


def _normalize_scheduler_name(name: Any) -> str:
    scheduler = str(name).lower()
    if scheduler == "step":
        return "multistep"
    return scheduler


def _config_limit(raw_config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = raw_config.get(key)
        if value is None:
            continue
        value = int(value)
        return None if value < 0 else value
    return None


def resolved_config(args: argparse.Namespace, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_config = raw_config or {}
    dataset_name = normalize_dataset_name(args.dataset or raw_config.get("dataset", "cifar10"))
    recipe_name = args.recipe or raw_config.get("recipe", "openmixup")
    method_name = normalize_method_name(args.method or raw_config.get("method", "fmix"))
    dataset = get_dataset_preset(dataset_name)
    recipe = get_recipe_preset(dataset_name, recipe_name)
    method_section = _section(raw_config, method_name)
    fmix_section = _section(raw_config, "fmix")
    guidedmixup_section = _section(raw_config, "guidedmixup") or _section(raw_config, "guided_sr")
    saliencymix_section = _section(raw_config, "saliencymix")

    config_augment = raw_config.get(
        "use_basic_augmentation",
        raw_config.get("augment", raw_config.get("basic_aug", True)),
    )
    use_basic_augmentation = bool(config_augment)
    if args.no_augment is True:
        use_basic_augmentation = False

    if args.mix_prob is not None:
        method_prob = args.mix_prob
    elif args.fmix_prob is not None:
        method_prob = args.fmix_prob
    else:
        method_prob = method_section.get(
            "prob",
            _first_config_value(
                raw_config,
                [
                    f"{method_name}_prob",
                    "guidedmixup_prob",
                    "saliencymix_prob",
                    "mix_prob",
                    "fmix_prob",
                ],
                1.0,
            ),
        )

    method_alpha = args.alpha
    if method_alpha is None:
        method_alpha = method_section.get(
            "alpha",
            _first_config_value(
                raw_config,
                [
                    f"{method_name}_alpha",
                    "guidedmixup_alpha",
                    "saliencymix_alpha",
                    "mixup_alpha",
                    "fmix_alpha",
                ],
                fmix_section.get("alpha", recipe.alpha),
            ),
        )

    config = {
        "dataset": dataset_name,
        "recipe": recipe_name,
        "model": raw_config.get("model", "preact_resnet18"),
        "method": method_name,
        "data_dir": args.data_dir or raw_config.get("data_dir", "./data"),
        "output_dir": args.output_dir or raw_config.get("output_dir", "./runs/fmix"),
        "checkpoint": args.checkpoint or raw_config.get("checkpoint") or raw_config.get("resume_checkpoint") or None,
        "download": bool(args.download if args.download is not None else raw_config.get("download", False)),
        "use_basic_augmentation": use_basic_augmentation,
        "num_classes": int(raw_config.get("num_classes", dataset.num_classes)),
        "image_size": dataset.image_size,
        "mean": dataset.mean,
        "std": dataset.std,
        "epochs": _choose(args, raw_config, "epochs", "training", "epochs", recipe.epochs),
        "batch_size": _choose(args, raw_config, "batch_size", "training", "batch_size", recipe.batch_size),
        "lr": _choose(args, raw_config, "lr", "training", "lr", raw_config.get("learning_rate", recipe.lr)),
        "momentum": _choose(args, raw_config, "momentum", "training", "momentum", recipe.momentum),
        "weight_decay": _choose(args, raw_config, "weight_decay", "training", "weight_decay", recipe.weight_decay),
        "scheduler": _normalize_scheduler_name(
            _choose(args, raw_config, "scheduler", "training", "scheduler", raw_config.get("lr_schedule", recipe.scheduler))
        ),
        "milestones": _choose(
            args,
            raw_config,
            "milestones",
            "training",
            "milestones",
            raw_config.get("lr_decay_epochs", list(recipe.milestones)),
        ),
        "lr_decay_rate": float(raw_config.get("lr_decay_rate", 0.1)),
        "min_learning_rate": float(raw_config.get("min_learning_rate", 0.0)),
        "alpha": method_alpha,
        "decay_power": _choose(args, raw_config, "decay_power", "fmix", "decay_power", recipe.decay_power),
        "max_soft": _choose(args, raw_config, "max_soft", "fmix", "max_soft", recipe.max_soft),
        "transform_profile": recipe.transform_profile,
        "reformulate": _choose(args, raw_config, "reformulate", "fmix", "reformulate", False),
        "method_prob": method_prob,
        "fmix_prob": method_prob,
        "guidedmixup_prob": method_prob,
        "saliencymix_prob": method_prob,
        "guidedmixup_blur_kernel": int(
            getattr(args, "guidedmixup_blur_kernel", None)
            if getattr(args, "guidedmixup_blur_kernel", None) is not None
            else guidedmixup_section.get("blur_kernel", raw_config.get("guidedmixup_blur_kernel", 7))
        ),
        "guidedmixup_condition": str(
            getattr(args, "guidedmixup_condition", None)
            or guidedmixup_section.get("condition")
            or raw_config.get("guidedmixup_condition", "greedy")
        ).lower(),
        "saliency_source": str(
            getattr(args, "saliency_source", None)
            or saliencymix_section.get("saliency_source")
            or raw_config.get("saliency_source", "spectral_residual")
        ).lower(),
        "saliency_dir": getattr(args, "saliency_dir", None) or raw_config.get("saliency_dir", raw_config.get("data_dir", "./data")),
        "saliency_path": getattr(args, "saliency_path", None) or raw_config.get("saliency_path") or None,
        "cross_device_shuffle": bool(raw_config.get("cross_device_shuffle", False)),
        "validation_split": float(raw_config.get("validation_split", 0.0)),
        "eval_on_test_each_epoch": bool(raw_config.get("eval_on_test_each_epoch", True)),
        "final_test": bool(raw_config.get("final_test", False)),
        "run_name": raw_config.get("run_name", ""),
        "save_csv": bool(raw_config.get("save_csv", False)),
        "output_name": raw_config.get("output_name", ""),
        "save_checkpoint": bool(raw_config.get("save_checkpoint", True)),
        "save_best_only": bool(raw_config.get("save_best_only", False)),
        "checkpoint_dir": raw_config.get("checkpoint_dir"),
    }
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_validation_split(
    train_set,
    val_set,
    preset,
    recipe,
    config: dict[str, Any],
    seed: int,
):
    split = float(config["validation_split"])
    if train_set is None or split <= 0:
        return train_set, val_set, None
    if split >= 1:
        raise ValueError(f"validation_split must be in [0, 1), got {split}.")

    eval_train_set, _ = build_datasets(
        preset,
        recipe.transform_profile,
        data_dir=config["data_dir"],
        download=bool(config["download"]),
        use_basic_augmentation=False,
    )

    num_total = len(train_set)
    num_val = max(1, int(round(num_total * split)))
    num_train = num_total - num_val
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_total, generator=generator).tolist()
    train_indices = permutation[:num_train]
    val_indices = permutation[num_train:]

    split_train_set = Subset(train_set, train_indices)
    split_val_set = Subset(eval_train_set, val_indices)
    test_set = val_set
    return split_train_set, split_val_set, test_set


def _xla_rank(xm: Any, xr: Any | None = None) -> int:
    if xr is not None and hasattr(xr, "global_ordinal"):
        return int(xr.global_ordinal())
    if hasattr(xm, "get_ordinal"):
        return int(xm.get_ordinal())
    return 0


def _xla_world_size(xm: Any, xr: Any | None = None) -> int:
    if xr is not None and hasattr(xr, "world_size"):
        return int(xr.world_size())
    if hasattr(xm, "xrt_world_size"):
        return int(xm.xrt_world_size())
    if hasattr(xm, "world_size"):
        return int(xm.world_size())
    return 1


def is_master(use_xla: bool, xm: Any | None, xr: Any | None = None) -> bool:
    if not use_xla:
        return True
    if xr is not None and hasattr(xr, "global_ordinal"):
        return int(xr.global_ordinal()) == 0
    if hasattr(xm, "is_master_ordinal"):
        return bool(xm.is_master_ordinal())
    return _xla_rank(xm, xr) == 0


def print_master(message: str, use_xla: bool, xm: Any | None, xr: Any | None = None) -> None:
    if use_xla:
        if hasattr(xm, "master_print"):
            xm.master_print(message)
        elif is_master(use_xla, xm, xr):
            print(message, flush=True)
    else:
        print(message, flush=True)


CSV_FIELDS = [
    "epoch",
    "phase",
    "lr",
    "train_loss",
    "train_accuracy",
    "train_top1",
    "train_top1_error",
    "eval_loss",
    "eval_top1_accuracy",
    "eval_top1_error",
    "val_loss",
    "val_top1",
    "val_top1_error",
    "best_top1_error",
    "best_epoch",
    "best_top1",
    "test_loss",
    "test_top1_accuracy",
    "test_top1_error",
    "test_top1",
]


def metrics_csv_path(run_dir: Path, config: dict[str, Any]) -> Path:
    output_name = str(config.get("output_name") or "").strip()
    filename = f"{output_name}.csv" if output_name else "metrics.csv"
    return run_dir / filename


def pct_to_fraction(value: float) -> float:
    return float(value) / 100.0


def pct_to_error(value: float) -> float:
    return 1.0 - pct_to_fraction(value)


def _maybe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_metrics_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in CSV_FIELDS}

    def fill_from_percent(percent_field: str, accuracy_field: str, error_field: str) -> None:
        percent = _maybe_float(row.get(percent_field))
        if percent is None:
            return
        if not normalized.get(accuracy_field):
            normalized[accuracy_field] = f"{pct_to_fraction(percent):.6f}"
        if not normalized.get(error_field):
            normalized[error_field] = f"{pct_to_error(percent):.6f}"

    def fill_error_from_accuracy(accuracy_field: str, error_field: str) -> None:
        accuracy = _maybe_float(normalized.get(accuracy_field) or row.get(accuracy_field))
        if accuracy is not None and not normalized.get(error_field):
            normalized[error_field] = f"{1.0 - accuracy:.6f}"

    fill_from_percent("train_top1", "train_accuracy", "train_top1_error")
    fill_from_percent("val_top1", "eval_top1_accuracy", "eval_top1_error")
    fill_from_percent("test_top1", "test_top1_accuracy", "test_top1_error")

    val_top1 = _maybe_float(row.get("val_top1"))
    if val_top1 is not None and not normalized.get("val_top1_error"):
        normalized["val_top1_error"] = f"{pct_to_error(val_top1):.6f}"

    best_top1 = _maybe_float(row.get("best_top1"))
    if best_top1 is not None and not normalized.get("best_top1_error"):
        normalized["best_top1_error"] = f"{pct_to_error(best_top1):.6f}"

    fill_error_from_accuracy("train_accuracy", "train_top1_error")
    fill_error_from_accuracy("eval_top1_accuracy", "eval_top1_error")
    fill_error_from_accuracy("test_top1_accuracy", "test_top1_error")

    return normalized


def migrate_metrics_csv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == CSV_FIELDS:
            return
        rows = [normalize_metrics_row(row) for row in reader]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    migrate_metrics_csv(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalize_metrics_row(row))


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> int:
    return int(topk_correct_tensor(logits, targets, k).item())


def topk_correct_tensor(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> torch.Tensor:
    _, pred = logits.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    return correct[:k].reshape(-1).float().sum()


def reduce_logits_for_dataset(logits: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset != "imagenet_a":
        return logits

    if logits.size(1) == IMAGENET_A_NUM_CLASSES:
        return logits
    if logits.size(1) != 1000:
        raise ValueError(
            "ImageNet-A evaluation expects either 1000 ImageNet logits or "
            f"{IMAGENET_A_NUM_CLASSES} already-reduced logits; got {logits.size(1)}."
        )

    indices = torch.tensor(IMAGENET_A_INDICES_IN_1K, device=logits.device)
    return logits.index_select(1, indices)


def reduce_metrics(metrics: tuple[float, int, int], name: str, use_xla: bool, xm: Any | None):
    if not use_xla:
        return metrics

    def _sum(values):
        return tuple(sum(value[i] for value in values) for i in range(3))

    return xm.mesh_reduce(name, metrics, _sum)


def _sum_tensors(values):
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def reduce_metric_tensors(
    loss_sum: torch.Tensor,
    correct: torch.Tensor,
    total: torch.Tensor,
    name: str,
    use_xla: bool,
    xm: Any | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_xla:
        return loss_sum, correct, total
    return (
        xm.mesh_reduce(f"{name}_loss", loss_sum, _sum_tensors),
        xm.mesh_reduce(f"{name}_correct", correct, _sum_tensors),
        xm.mesh_reduce(f"{name}_total", total, _sum_tensors),
    )


def _tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _log_xla_progress(
    loss_sum: torch.Tensor,
    correct: torch.Tensor,
    total: torch.Tensor,
    epoch: int,
    step: int,
    start: float,
) -> None:
    total_value = max(_tensor_float(total), 1.0)
    elapsed = time.time() - start
    print(
        f"epoch={epoch} step={step} loss={_tensor_float(loss_sum) / total_value:.4f} "
        f"top1={100.0 * _tensor_float(correct) / total_value:.2f} imgs/s={total_value / max(elapsed, 1e-9):.1f}",
        flush=True,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_acc: float,
    config: dict[str, Any],
    use_xla: bool,
    xm: Any | None,
) -> None:
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_xla:
        xm.save(payload, str(path))
    else:
        torch.save(payload, path)


def load_model_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {path} does not contain a model state dict.")

    state_dict = prepare_state_dict_for_model(state_dict, model)

    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def prepare_state_dict_for_model(state_dict: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}

    model_keys = set(model.state_dict())
    if model_keys.intersection(state_dict):
        return state_dict

    if "fc.weight" not in state_dict and "fc.bias" not in state_dict:
        return state_dict

    mapped = {}
    for key, value in state_dict.items():
        key = str(key)
        if key.startswith("fc."):
            mapped[f"head.{key}"] = value
        else:
            mapped[f"backbone.{key}"] = value
    return mapped if model_keys.intersection(mapped) else state_dict


def make_scheduler(optimizer: optim.Optimizer, config: dict[str, Any]):
    if config["scheduler"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["epochs"]),
            eta_min=float(config["min_learning_rate"]),
        )
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(config["milestones"]),
        gamma=float(config["lr_decay_rate"]),
    )


def build_batch_mixer(config: dict[str, Any]):
    method = str(config["method"]).lower()
    if method == "fmix":
        return FMix(
            decay_power=float(config["decay_power"]),
            alpha=float(config["alpha"]),
            size=(int(config["image_size"]), int(config["image_size"])),
            max_soft=float(config["max_soft"]),
            reformulate=bool(config["reformulate"]),
        )
    if method == "mixup":
        return MixUp(alpha=float(config["alpha"]))
    if method == "saliencymix":
        return SaliencyMix(
            alpha=float(config["alpha"]),
            saliency_source=str(config.get("saliency_source", "spectral_residual")),
            blur_kernel=int(config.get("guidedmixup_blur_kernel", 7)),
        )
    if method == "guided_sr":
        return GuidedSR(
            alpha=float(config["alpha"]),
            blur_kernel=int(config.get("guidedmixup_blur_kernel", 7)),
            condition=str(config.get("guidedmixup_condition", "greedy")),
        )
    if method in {"baseline", "none", "eval"}:
        return None
    raise ValueError(f"Unsupported method: {config['method']}")


def mixed_sample_cross_entropy(logits: torch.Tensor, mixed, config: dict[str, Any]) -> torch.Tensor:
    method = str(config["method"]).lower()
    if method == "fmix":
        return fmix_cross_entropy(
            logits,
            mixed.targets_a,
            mixed.targets_b,
            mixed.lam,
            reformulate=bool(config["reformulate"]),
        )
    if method == "mixup":
        return mixup_cross_entropy(logits, mixed.targets_a, mixed.targets_b, mixed.lam)
    if method in {"saliencymix", "guided_sr"}:
        return mixup_cross_entropy(logits, mixed.targets_a, mixed.targets_b, mixed.lam)
    raise ValueError(f"Unsupported mixed-sample method: {config['method']}")


def cross_device_shuffle_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    rank: int,
    xm: Any,
    aux_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local_batch = int(images.size(0))
    global_images = xm.all_gather(images, dim=0)
    global_targets = xm.all_gather(targets, dim=0)
    global_aux = xm.all_gather(aux_tensor, dim=0) if aux_tensor is not None else None

    local_scores = torch.rand(local_batch, device=images.device)
    global_scores = xm.all_gather(local_scores, dim=0)
    global_index = torch.argsort(global_scores)

    start = rank * local_batch
    partner_index = global_index.narrow(0, start, local_batch)
    partner_images = global_images.index_select(0, partner_index)
    partner_targets = global_targets.index_select(0, partner_index)
    if global_aux is not None:
        partner_aux = global_aux.index_select(0, partner_index)
        return partner_images, partner_targets, partner_index, partner_aux
    return partner_images, partner_targets, partner_index


def unpack_train_batch(batch) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if not isinstance(batch, (tuple, list)):
        raise ValueError("train batch must be (images, targets) or (images, targets, saliency_maps).")
    if len(batch) == 2:
        images, targets = batch
        return images, targets, {}
    if len(batch) >= 3:
        images, targets, saliency_maps = batch[:3]
        return images, targets, {"saliency_maps": saliency_maps}
    raise ValueError("train batch must be (images, targets) or (images, targets, saliency_maps).")


def call_batch_mixer(
    mixer,
    images: torch.Tensor,
    targets: torch.Tensor,
    aux_info: dict[str, torch.Tensor],
    config: dict[str, Any],
    use_xla: bool,
    xm: Any | None,
    rank: int,
    world_size: int,
):
    method = str(config["method"]).lower()
    saliency_maps = aux_info.get("saliency_maps")

    if method == "saliencymix":
        if use_xla and bool(config["cross_device_shuffle"]) and world_size > 1:
            if saliency_maps is not None:
                partner_images, partner_targets, partner_index, partner_saliency_maps = cross_device_shuffle_batch(
                    images,
                    targets,
                    rank,
                    xm,
                    aux_tensor=saliency_maps,
                )
                return mixer(
                    images,
                    targets,
                    saliency_maps=saliency_maps,
                    partner_images=partner_images,
                    partner_targets=partner_targets,
                    partner_saliency_maps=partner_saliency_maps,
                    index=partner_index,
                )
            partner_images, partner_targets, partner_index = cross_device_shuffle_batch(images, targets, rank, xm)
            return mixer(
                images,
                targets,
                partner_images=partner_images,
                partner_targets=partner_targets,
                index=partner_index,
            )
        return mixer(images, targets, saliency_maps=saliency_maps)

    if method == "guided_sr":
        return mixer(images, targets, saliency_maps=saliency_maps)

    if use_xla and bool(config["cross_device_shuffle"]) and world_size > 1:
        partner_images, partner_targets, partner_index = cross_device_shuffle_batch(images, targets, rank, xm)
        return mixer(images, targets, partner_images, partner_targets, partner_index)
    return mixer(images, targets)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: optim.Optimizer,
    mixer,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    use_xla: bool,
    xm: Any | None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[float, float]:
    model.train()
    if use_xla:
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        total = torch.zeros((), device=device)
    else:
        loss_sum = 0.0
        correct = 0
        total = 0
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        if args.max_train_steps is not None and step > args.max_train_steps:
            break
        images, targets, aux_info = unpack_train_batch(batch)
        if not use_xla:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            aux_info = {key: value.to(device, non_blocking=True) for key, value in aux_info.items()}

        optimizer.zero_grad(set_to_none=True)
        if mixer is not None and config["method_prob"] > 0 and random.random() < config["method_prob"]:
            mixed = call_batch_mixer(mixer, images, targets, aux_info, config, use_xla, xm, rank, world_size)
            logits = model(mixed.images)
            loss = mixed_sample_cross_entropy(logits, mixed, config)
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        loss.backward()
        if use_xla:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()

        batch_size = int(images.size(0))
        if use_xla:
            batch_total = torch.tensor(float(batch_size), device=device)
            loss_sum = loss_sum + loss.detach() * batch_total
            correct = correct + topk_correct_tensor(logits.detach(), targets, k=1)
            total = total + batch_total
        else:
            loss_sum += float(loss.item()) * batch_size
            correct += topk_correct(logits.detach(), targets, k=1)
            total += batch_size

        if args.log_interval and step % args.log_interval == 0:
            if use_xla:
                if is_master(use_xla, xm):
                    xm.add_step_closure(
                        _log_xla_progress,
                        args=(loss_sum, correct, total, epoch, step, start),
                        run_async=True,
                    )
            else:
                elapsed = time.time() - start
                print_master(
                    f"epoch={epoch} step={step} loss={loss_sum / max(total, 1):.4f} "
                    f"top1={100.0 * correct / max(total, 1):.2f} imgs/s={total / max(elapsed, 1e-9):.1f}",
                    use_xla,
                    xm,
                )

    if use_xla:
        xm.mark_step()
        loss_sum, correct, total = reduce_metric_tensors(loss_sum, correct, total, f"train_{epoch}", use_xla, xm)
        total_value = max(_tensor_float(total), 1.0)
        return _tensor_float(loss_sum) / total_value, 100.0 * _tensor_float(correct) / total_value

    loss_sum, correct, total = reduce_metrics((loss_sum, correct, total), f"train_{epoch}", use_xla, xm)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    use_xla: bool,
    xm: Any | None,
) -> tuple[float, float]:
    model.eval()
    if use_xla:
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        total = torch.zeros((), device=device)
    else:
        loss_sum = 0.0
        correct = 0
        total = 0

    for step, (images, targets) in enumerate(loader, start=1):
        if args.max_val_steps is not None and step > args.max_val_steps:
            break
        if not use_xla:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        logits = reduce_logits_for_dataset(model(images), str(config["dataset"]))
        loss = F.cross_entropy(logits, targets)
        batch_size = int(images.size(0))
        if use_xla:
            batch_total = torch.tensor(float(batch_size), device=device)
            loss_sum = loss_sum + loss.detach() * batch_total
            correct = correct + topk_correct_tensor(logits, targets, k=1)
            total = total + batch_total
        else:
            loss_sum += float(loss.item()) * batch_size
            correct += topk_correct(logits, targets, k=1)
            total += batch_size
        if use_xla:
            xm.mark_step()

    if use_xla:
        xm.mark_step()
        loss_sum, correct, total = reduce_metric_tensors(loss_sum, correct, total, f"val_{epoch}", use_xla, xm)
        total_value = max(_tensor_float(total), 1.0)
        return _tensor_float(loss_sum) / total_value, 100.0 * _tensor_float(correct) / total_value

    loss_sum, correct, total = reduce_metrics((loss_sum, correct, total), f"val_{epoch}", use_xla, xm)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


def run_worker(index: int, args: argparse.Namespace) -> None:
    xla_modules = _optional_xla_import()
    requested_xla = args.device == "xla"
    use_xla = bool(requested_xla)
    if requested_xla and xla_modules is None:
        raise RuntimeError("PyTorch/XLA is not installed. Install torch_xla or use --device cpu/cuda.")

    xm = xla_modules["xm"] if use_xla else None
    pl = xla_modules["pl"] if use_xla else None
    xr = xla_modules["xr"] if use_xla else None

    if use_xla:
        device = xm.xla_device()
        rank = _xla_rank(xm, xr)
        world_size = _xla_world_size(xm, xr)
    elif args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        rank = 0
        world_size = 1
    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1

    raw_config = load_config(args.config)
    if args.seed is None:
        args.seed = int(raw_config.get("seed", 1))
    set_seed(args.seed + rank)
    if args.max_train_steps is None:
        args.max_train_steps = _config_limit(raw_config, "max_train_steps")
    if args.max_val_steps is None:
        args.max_val_steps = _config_limit(raw_config, "max_val_steps", "max_eval_steps")
    config = resolved_config(args, raw_config)
    preset = get_dataset_preset(config["dataset"])
    recipe = get_recipe_preset(config["dataset"], config["recipe"])

    train_set, val_set = build_datasets(
        preset,
        recipe.transform_profile,
        data_dir=config["data_dir"],
        download=bool(config["download"]),
        use_basic_augmentation=bool(config["use_basic_augmentation"]),
    )
    if train_set is not None and str(config["method"]) == "saliencymix" and str(config["saliency_source"]) == "batch":
        train_set = attach_train_saliency_maps(
            train_set,
            dataset_name=str(config["dataset"]),
            saliency_dir=str(config["saliency_dir"]),
            saliency_path=config["saliency_path"],
        )
    train_set, val_set, test_set = apply_validation_split(train_set, val_set, preset, recipe, config, args.seed)
    if bool(config["eval_on_test_each_epoch"]) and test_set is not None:
        val_set = test_set

    distributed = world_size > 1
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        if train_set is not None and distributed
        else None
    )
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    test_sampler = (
        DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
        if test_set is not None and distributed
        else None
    )

    train_loader = None
    if train_set is not None:
        train_loader = DataLoader(
            train_set,
            batch_size=int(config["batch_size"]),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    if use_xla:
        if train_loader is not None:
            train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)
        if test_loader is not None:
            test_loader = pl.MpDeviceLoader(test_loader, device)

    model = build_model(str(config["model"]), num_classes=int(config["num_classes"]))
    if config["checkpoint"]:
        checkpoint_meta = load_model_checkpoint(str(config["checkpoint"]), model)
        loaded_epoch = checkpoint_meta.get("epoch")
        suffix = f" at epoch {loaded_epoch}" if loaded_epoch is not None else ""
        print_master(f"Loaded checkpoint {config['checkpoint']}{suffix}", use_xla, xm)
    model = model.to(device)

    if config["run_name"]:
        run_dir = Path(config["output_dir"]) / str(config["run_name"])
    else:
        run_dir = Path(config["output_dir"]) / config["dataset"] / config["recipe"]
    checkpoint_root = (
        Path(config["checkpoint_dir"]) / str(config["run_name"])
        if config["checkpoint_dir"] and config["run_name"]
        else Path(config["checkpoint_dir"])
        if config["checkpoint_dir"]
        else run_dir
    )
    if is_master(use_xla, xm, xr):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "resolved": config,
                    "preset": preset_dict(config["dataset"], config["recipe"]),
                    "args": vars(args),
                },
                indent=2,
            )
        )
    csv_path = metrics_csv_path(run_dir, config)
    print_master(
        f"Starting {str(config['method']).upper()} {config['dataset']}/{config['recipe']} on {device}; "
        f"world_size={world_size}; config={json.dumps(config, sort_keys=True)}",
        use_xla,
        xm,
        xr,
    )

    if train_loader is None or int(config["epochs"]) <= 0:
        if train_loader is None and not config["checkpoint"]:
            print_master(
                "No train split is available and no checkpoint was provided; evaluating randomly initialized weights.",
                use_xla,
                xm,
                xr,
            )
        val_loss, val_acc = evaluate(model, val_loader, device, 0, config, args, use_xla, xm)
        if bool(config["save_csv"]) and is_master(use_xla, xm, xr):
            val_error = pct_to_error(val_acc)
            append_metrics_csv(
                csv_path,
                {
                    "epoch": 0,
                    "phase": "eval",
                    "eval_loss": f"{val_loss:.6f}",
                    "eval_top1_accuracy": f"{pct_to_fraction(val_acc):.6f}",
                    "eval_top1_error": f"{val_error:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_top1": f"{val_acc:.6f}",
                    "val_top1_error": f"{val_error:.6f}",
                    "best_top1": f"{val_acc:.6f}",
                    "best_top1_error": f"{val_error:.6f}",
                    "best_epoch": 0,
                },
            )
        print_master(f"eval_loss={val_loss:.4f} eval_top1={val_acc:.2f}", use_xla, xm, xr)
        return

    optimizer = optim.SGD(
        model.parameters(),
        lr=float(config["lr"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, config)
    mixer = build_batch_mixer(config)

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, mixer, device, epoch, config, args, use_xla, xm, rank, world_size
        )
        val_loss, val_acc = evaluate(model, val_loader, device, epoch, config, args, use_xla, xm)
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            if bool(config["save_checkpoint"]) and is_master(use_xla, xm, xr):
                save_checkpoint(checkpoint_root / "best.pt", model, optimizer, scheduler, epoch, best_acc, config, use_xla, xm)
        if (
            bool(config["save_checkpoint"])
            and not bool(config["save_best_only"])
            and args.save_every
            and epoch % args.save_every == 0
            and is_master(use_xla, xm, xr)
        ):
            save_checkpoint(
                checkpoint_root / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                config,
                use_xla,
                xm,
            )
        if bool(config["save_csv"]) and is_master(use_xla, xm, xr):
            train_error = pct_to_error(train_acc)
            val_error = pct_to_error(val_acc)
            best_error = pct_to_error(best_acc)
            append_metrics_csv(
                csv_path,
                {
                    "epoch": epoch,
                    "phase": "train_val",
                    "lr": f"{epoch_lr:.10g}",
                    "train_loss": f"{train_loss:.6f}",
                    "train_accuracy": f"{pct_to_fraction(train_acc):.6f}",
                    "train_top1": f"{train_acc:.6f}",
                    "train_top1_error": f"{train_error:.6f}",
                    "eval_loss": f"{val_loss:.6f}",
                    "eval_top1_accuracy": f"{pct_to_fraction(val_acc):.6f}",
                    "eval_top1_error": f"{val_error:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_top1": f"{val_acc:.6f}",
                    "val_top1_error": f"{val_error:.6f}",
                    "best_top1_error": f"{best_error:.6f}",
                    "best_epoch": best_epoch,
                    "best_top1": f"{best_acc:.6f}",
                },
            )

        print_master(
            f"epoch={epoch} train_loss={train_loss:.4f} train_top1={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_top1={val_acc:.2f} best_top1={best_acc:.2f}",
            use_xla,
            xm,
            xr,
        )

    if bool(config["save_checkpoint"]) and not bool(config["save_best_only"]) and is_master(use_xla, xm, xr):
        save_checkpoint(checkpoint_root / "last.pt", model, optimizer, scheduler, int(config["epochs"]), best_acc, config, use_xla, xm)
    if bool(config["final_test"]) and test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, device, int(config["epochs"]), config, args, use_xla, xm)
        if bool(config["save_csv"]) and is_master(use_xla, xm, xr):
            test_error = pct_to_error(test_acc)
            append_metrics_csv(
                csv_path,
                {
                    "epoch": int(config["epochs"]),
                    "phase": "final_test",
                    "best_top1": f"{best_acc:.6f}",
                    "best_top1_error": f"{pct_to_error(best_acc):.6f}",
                    "best_epoch": best_epoch,
                    "test_loss": f"{test_loss:.6f}",
                    "test_top1_accuracy": f"{pct_to_fraction(test_acc):.6f}",
                    "test_top1_error": f"{test_error:.6f}",
                    "test_top1": f"{test_acc:.6f}",
                },
            )
        print_master(f"final_test_loss={test_loss:.4f} final_test_top1={test_acc:.2f}", use_xla, xm, xr)
    print_master(f"Finished. best_top1={best_acc:.2f}", use_xla, xm, xr)


def main() -> None:
    args = parse_args()
    if args.device == "xla" and args.num_cores > 1:
        os.environ.setdefault("TPU_NUM_DEVICES", str(args.num_cores))
    should_spawn = args.device == "xla" and args.num_cores > 1
    if should_spawn:
        launcher = _optional_xla_launcher()
        if launcher is None:
            raise RuntimeError("PyTorch/XLA is not installed. Cannot spawn XLA workers.")
        launcher(run_worker, args=(args,), start_method="spawn")
    else:
        run_worker(0, args)


if __name__ == "__main__":
    main()
