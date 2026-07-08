"""Summarize experiment metrics into paper-table friendly rows."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from allthemix.data.saliency_dataset import saliency_array_is_finite, saliency_path_candidates
from allthemix.cli.presets import (
    DATASET_EXPECTED_SPLIT_COUNTS,
    get_dataset_preset,
    normalize_dataset_name,
)
from allthemix.networks import canonical_model_name
from allthemix.cli.train import (
    _as_bool,
    default_guided_sr_saliency_path,
    load_config,
    normalize_method_name,
    parse_args as parse_train_args,
    relocate_relative_saliency_path,
    resolve_saliency_storage_paths,
    resolved_config,
)


@dataclass(frozen=True)
class ExperimentSpec:
    type_name: str
    method_label: str
    method_key: str
    config_path: Path
    script_path: Path | None = None


@dataclass(frozen=True)
class ExperimentSummary:
    spec: ExperimentSpec
    metrics_path: Path
    error: float | None
    metric_source: str
    status: str
    prerequisite_status: str = "ok"
    prerequisite_path: Path | None = None
    resume_checkpoint_path: Path | None = None
    best_checkpoint_path: Path | None = None
    final_test_checkpoint: str = ""
    final_test_checkpoint_source: str = ""


@dataclass(frozen=True)
class ProtocolIssue:
    method_key: str
    config_path: Path
    field: str
    expected: object
    actual: object


class TrainArgValidationError(ValueError):
    pass


def _tiny_xla4_spec(type_name: str, method_label: str, method_key: str) -> ExperimentSpec:
    return ExperimentSpec(
        type_name,
        method_label,
        method_key,
        Path(f"configs/tiny_imagenet/preact_resnet18/{method_key}_xla4.yaml"),
        Path(f"scripts/experiment_run/run_tiny_imagenet_preact_resnet18_{method_key}_xla4.sh"),
    )


TINY_IMAGENET_XLA4_SPECS = [
    _tiny_xla4_spec("Baseline", "ERM", "baseline"),
    _tiny_xla4_spec("MixDA", "MixUp", "mixup"),
    _tiny_xla4_spec("MixDA", "CutMix", "cutmix"),
    _tiny_xla4_spec("MixDA", "ResizeMix", "resizemix"),
    _tiny_xla4_spec("MixDA", "FMix", "fmix"),
    _tiny_xla4_spec("MixDA", "SaliencyMix", "saliencymix"),
    _tiny_xla4_spec("MixDA", "Guided-SR", "guided_sr"),
    _tiny_xla4_spec("MixDA", "CatchUpMix", "catchupmix"),
]


METHOD_FILTER_ALIASES = {
    "erm": "baseline",
    "guidedmixup": "guided_sr",
    "guided_mixup": "guided_sr",
    "guided_mixup_sr": "guided_sr",
    "guidedsr": "guided_sr",
    "saliency_mix": "saliencymix",
    "resize_mix": "resizemix",
    "catchup_mix": "catchupmix",
    "catch_up_mix": "catchupmix",
}


def _normalized_method_filter(value: str) -> str:
    normalized = normalize_method_name(value.lower().replace("-", "_").replace(" ", "_"))
    return METHOD_FILTER_ALIASES.get(normalized, normalized)


def filter_specs_by_method(
    specs: Iterable[ExperimentSpec],
    selected_methods: Iterable[str] = (),
) -> list[ExperimentSpec]:
    selected = list(selected_methods)
    specs_list = list(specs)
    if not selected:
        return specs_list

    aliases: dict[str, str] = {}
    for spec in specs_list:
        aliases[_normalized_method_filter(spec.method_key)] = spec.method_key
        aliases[_normalized_method_filter(spec.method_label)] = spec.method_key

    wanted = []
    unknown = []
    for method in selected:
        normalized = _normalized_method_filter(method)
        method_key = aliases.get(normalized, normalized)
        if method_key not in {spec.method_key for spec in specs_list}:
            unknown.append(method)
        elif method_key not in wanted:
            wanted.append(method_key)
    if unknown:
        available = ", ".join(spec.method_key for spec in specs_list)
        raise ValueError(f"Unknown method filter: {', '.join(unknown)}. Available methods: {available}")

    return [spec for spec in specs_list if spec.method_key in wanted]


TINY_IMAGENET_XLA4_PROTOCOL_ID = "allthemix_split200_openmixup_aug_bestval_xla4"
TINY_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
TINY_IMAGENET_EXPECTED_COUNTS = dict(DATASET_EXPECTED_SPLIT_COUNTS["tinyimagenet"])


TINY_IMAGENET_XLA4_PROTOCOL = {
    "id": TINY_IMAGENET_XLA4_PROTOCOL_ID,
    "purpose": "Tiny-ImageNet table runs for AllTheMiXLA.",
    "dataset": "Tiny-ImageNet",
    "model": "PreAct-ResNet-18",
    "epochs": 200,
    "per_device_batch_size": 32,
    "expected_tpu_devices": 4,
    "global_batch_size": 128,
    "optimizer": "SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)",
    "scheduler": "MultiStepLR(milestones=[150, 180], gamma=0.1)",
    "split": "validation_split=0.1, eval_on_test_each_epoch=false, final_test=true, final_test_checkpoint=best",
    "train_transform": "RandomResizedCrop(64, bicubic) + RandomHorizontalFlip + ImageNet normalization",
    "saliency_transform": "cached saliency maps receive the same paired Tiny-OpenMixup spatial transform before normalization",
    "guided_sr_reference": "GuidedMixup SR-style method settings: prob=0.5, condition=greedy, and online spectral-residual saliency computed after denormalizing augmented tensors back to unit image space and per-image min-max normalization; alpha is kept for the shared MixDA interface.",
    "openmixup_reference": "OpenMixup Tiny-ImageNet benchmark uses 400 epochs, global batch 100, lr=0.2, cosine schedule, and reports last-10-epoch median accuracy.",
}


MAIN_TABLE_ROWS = [
    ("Baseline", "ERM", "baseline", "4.94", "24.17", "14.31", "21.35", "--"),
    ("MixDA", "MixUp", "mixup", "4.11", "21.65", r"\textbf{10.79}", "20.54", "--"),
    ("MixDA", "CutMix", "cutmix", "3.62", "21.07", "11.69", "21.17", "--"),
    ("MixDA", "ResizeMix", "resizemix", r"\textbf{3.53}", r"\textbf{20.50}", "10.91", r"\textbf{17.26}", "--"),
    ("MixDA", "FMix", "fmix", "3.70", "20.71", "11.99", "--", "--"),
    ("MixDA", "SaliencyMix", "saliencymix", "3.69", "20.91", "--", "18.02", "--"),
    ("MixDA", "Guided-SR", "guided_sr", "4.31", "23.34", "12.34", "19.84", "--"),
    ("MixDA", "CatchUpMix", "catchupmix", "4.15", "20.55", "11.22", "--", "--"),
    (r"Baseline$^\dagger$", "ERM (split)", None, "5.22", "25.19", "--", "--", "--"),
    (r"Val-Aware$^\dagger$", "MetaAugment", None, "4.34", "23.23", "--", "--", "--"),
    (r"Val-Aware$^\dagger$", "IF-AugNet", None, "5.04", "23.95", "--", "--", "--"),
    (r"Baseline$^\ddagger$", "ERM", None, "--", "--", "--", "--", "--"),
    (r"Generative$^\ddagger$", "DA-Fusion", None, "--", "--", "--", "--", "--"),
]


COMMON_TINY_XLA4_EXPECTED = {
    "dataset": "tiny_imagenet",
    "data_dir": "./data",
    "model": "preact_resnet18",
    "batch_size": 32,
    "global_batch_size": 128,
    "epochs": 200,
    "max_train_steps": -1,
    "max_eval_steps": -1,
    "validation_split": 0.1,
    "eval_on_test_each_epoch": False,
    "final_test": True,
    "final_test_checkpoint": "best",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_schedule": "step",
    "min_learning_rate": 0.0,
    "lr_decay_epochs": [150, 180],
    "lr_decay_rate": 0.1,
    "save_csv": True,
    "run_metadata_required": True,
    "output_dir": "./outputs",
    "output_name": "",
    "save_checkpoint": True,
    "checkpoint_dir": "./checkpoints",
    "save_best_only": True,
    "resume_checkpoint": "",
    "distributed": True,
    "log_time": True,
    "seed": 0,
}


COMMON_TINY_XLA4_RESOLVED_EXPECTED = {
    "dataset": "tinyimagenet",
    "recipe": "openmixup",
    "model": "preact_resnet18",
    "model_impl_version": 2,
    "batch_size": 32,
    "global_batch_size": 128,
    "epochs": 200,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "scheduler": "multistep",
    "milestones": [150, 180],
    "lr_decay_rate": 0.1,
    "min_learning_rate": 0.0,
    "transform_profile": "openmixup",
    "num_classes": 200,
    "image_size": 64,
    "validation_split": 0.1,
    "eval_on_test_each_epoch": False,
    "final_test": True,
    "final_test_checkpoint": "best",
    "save_csv": True,
    "run_metadata_required": True,
    "output_dir": "./outputs",
    "output_name": "",
    "save_checkpoint": True,
    "checkpoint_dir": "./checkpoints",
    "save_best_only": True,
}


METHOD_TINY_XLA4_EXPECTED = {
    "baseline": {"method": "baseline", "basic_aug": False, "aug_recipe": "tiny_openmixup"},
    "mixup": {
        "method": "mixup",
        "cross_device_shuffle": True,
        "mixup_no_repeat": False,
        "basic_aug": False,
        "aug_recipe": "tiny_openmixup",
    },
    "cutmix": {
        "method": "cutmix",
        "cross_device_shuffle": True,
        "cutmix_alpha": 1.0,
        "cutmix_prob": 1.0,
        "cutmix_no_repeat": True,
        "basic_aug": False,
        "aug_recipe": "tiny_openmixup",
    },
    "resizemix": {
        "method": "resizemix",
        "cross_device_shuffle": True,
        "resizemix_scope_min": 0.1,
        "resizemix_scope_max": 0.4,
        "resizemix_use_alpha": False,
        "resizemix_prob": 1.0,
        "resizemix_no_repeat": False,
        "basic_aug": False,
        "aug_recipe": "tiny_openmixup",
    },
    "fmix": {"method": "fmix", "cross_device_shuffle": True, "basic_aug": False, "aug_recipe": "tiny_openmixup"},
    "saliencymix": {
        "method": "saliencymix",
        "cross_device_shuffle": True,
        "saliencymix_alpha": 1.0,
        "saliencymix_prob": 0.5,
        "saliencymix_no_repeat": False,
        "saliency_source": "batch",
        "saliency_dir": "./data",
        "sal_basic_aug": False,
        "sal_aug_recipe": "tiny_openmixup",
        "basic_aug": False,
    },
    "guided_sr": {
        "method": "guided_sr",
        "cross_device_shuffle": False,
        "guidedmixup_alpha": 1.0,
        "guidedmixup_prob": 0.5,
        "guidedmixup_blur_kernel": 7,
        "guidedmixup_condition": "greedy",
        "saliency_source": "spectral_residual",
        "basic_aug": False,
        "aug_recipe": "tiny_openmixup",
    },
    "catchupmix": {
        "method": "catchupmix",
        "catchupmix_alpha": 1.0,
        "catchupmix_cutmix_alpha": 1.0,
        "catchupmix_num_layers": 5,
        "catchupmix_no_repeat": False,
        "basic_aug": False,
        "aug_recipe": "tiny_openmixup",
    },
}


METHOD_TINY_XLA4_RESOLVED_EXPECTED = {
    "baseline": {
        "method": "baseline",
        "method_prob": 1.0,
        "alpha": 1.0,
        "use_basic_augmentation": False,
        "aug_recipe": "tiny_openmixup",
        "cross_device_shuffle": False,
    },
    "mixup": {
        "method": "mixup",
        "method_prob": 1.0,
        "alpha": 1.0,
        "use_basic_augmentation": False,
        "aug_recipe": "tiny_openmixup",
        "cross_device_shuffle": True,
        "mixup_no_repeat": False,
    },
    "cutmix": {
        "method": "cutmix",
        "method_prob": 1.0,
        "alpha": 1.0,
        "use_basic_augmentation": False,
        "aug_recipe": "tiny_openmixup",
        "cross_device_shuffle": True,
        "cutmix_no_repeat": True,
    },
    "resizemix": {
        "method": "resizemix",
        "method_prob": 1.0,
        "alpha": 1.0,
        "use_basic_augmentation": False,
        "aug_recipe": "tiny_openmixup",
        "cross_device_shuffle": True,
        "resizemix_scope_min": 0.1,
        "resizemix_scope_max": 0.4,
        "resizemix_use_alpha": False,
        "resizemix_no_repeat": False,
    },
    "fmix": {
        "method": "fmix",
        "method_prob": 1.0,
        "alpha": 1.0,
        "decay_power": 3.0,
        "max_soft": 0.0,
        "reformulate": False,
        "use_basic_augmentation": False,
        "aug_recipe": "tiny_openmixup",
        "cross_device_shuffle": True,
        "fmix_no_repeat": False,
    },
    "saliencymix": {
        "method": "saliencymix",
        "method_prob": 0.5,
        "alpha": 1.0,
        "use_basic_augmentation": False,
        "aug_recipe": None,
        "saliency_source": "batch",
        "saliency_dir": "./data",
        "saliency_path": None,
        "sal_basic_aug": False,
        "sal_aug_recipe": "tiny_openmixup",
        "cross_device_shuffle": True,
        "saliencymix_no_repeat": False,
    },
    "guided_sr": {
        "method": "guided_sr",
        "method_prob": 0.5,
        "alpha": 1.0,
        "use_basic_augmentation": False,
        "aug_recipe": "tiny_openmixup",
        "guidedmixup_blur_kernel": 7,
        "guidedmixup_condition": "greedy",
        "saliency_source": "spectral_residual",
        "saliency_dir": "./data",
        "saliency_path": None,
        "sal_basic_aug": False,
        "sal_aug_recipe": "none",
        "cross_device_shuffle": False,
    },
    "catchupmix": {
        "method": "catchupmix",
        "method_prob": 1.0,
        "alpha": 1.0,
        "use_basic_augmentation": False,
        "aug_recipe": "tiny_openmixup",
        "cross_device_shuffle": False,
        "catchupmix_cutmix_alpha": 1.0,
        "catchupmix_num_layers": 5,
        "catchupmix_no_repeat": False,
    },
}


FMIX_SECTION_EXPECTED = {
    "alpha": 1.0,
    "decay_power": 3.0,
    "max_soft": 0.0,
    "reformulate": False,
    "prob": 1.0,
    "no_repeat": False,
}
MIXUP_SECTION_EXPECTED = {"alpha": 1.0, "prob": 1.0}
EXPECTED_TRAIN_EXAMPLES = {
    dataset_name: counts["train"]
    for dataset_name, counts in DATASET_EXPECTED_SPLIT_COUNTS.items()
    if "train" in counts
}
EXPECTED_SALIENCY_SPATIAL = {
    "tinyimagenet": (64, 64),
}
SALIENCY_CACHE_MIN_VERSION = 4
EXPECTED_SALIENCY_CACHE_METHODS = {
    "saliencymix": {"opencv", "opencv_finegrained", "finegrained"},
    "guided_sr": {"spectral_residual", "sr", "guided_sr", "online"},
}
CACHE_COMMAND_FORWARD_VALUE_ARGS = {
    "--data-dir",
    "--recipe",
    "--saliency-dir",
    "--seed",
}
CACHE_COMMAND_RENAMED_VALUE_ARGS = {
    "--guidedmixup-blur-kernel": "--blur-kernel",
    "--saliency-path": "--output",
}
METHOD_SPECIFIC_TRAIN_VALUE_ARGS = {
    "--alpha": {"mixup", "cutmix", "resizemix", "fmix", "saliencymix", "guided_sr", "catchupmix"},
    "--decay-power": {"fmix"},
    "--fmix-prob": {"fmix"},
    "--guidedmixup-blur-kernel": {"guided_sr"},
    "--guidedmixup-condition": {"guided_sr"},
    "--max-soft": {"fmix"},
    "--mix-prob": {"mixup", "cutmix", "resizemix", "fmix", "saliencymix", "guided_sr", "catchupmix"},
    "--sal-aug-recipe": {"saliencymix", "guided_sr"},
    "--saliency-dir": {"saliencymix", "guided_sr"},
    "--saliency-path": {"saliencymix", "guided_sr"},
    "--saliency-source": {"saliencymix", "guided_sr"},
}
METHOD_SPECIFIC_TRAIN_FLAG_ARGS = {
    "--reformulate": {"fmix"},
}
RESERVED_GENERATED_COMMAND_TRAIN_ARGS = {
    "--config",
    "--device",
    "--dataset",
    "--num-cores",
    "--num-workers",
    "--output-dir",
}
RESERVED_GENERATED_COMMAND_TRAIN_ARG_HINTS = {
    "--config": "the summary preset controls each method config path",
    "--device": "use summarize's --device option instead",
    "--dataset": "the tiny-imagenet-xla4 preset controls the dataset; use --data-dir for data location overrides",
    "--num-cores": "use summarize's --num-cores option instead",
    "--num-workers": "use summarize's --num-workers option instead",
    "--output-dir": "generated summaries read the output_dir declared in each config; edit the YAML for persistent output relocation",
}
AUTO_RESUME_UNSAFE_TRAIN_ARGS = {
    "--alpha",
    "--aug-recipe",
    "--batch-size",
    "--checkpoint-dir",
    "--decay-power",
    "--dataset",
    "--epochs",
    "--final-test-checkpoint",
    "--fmix-prob",
    "--guidedmixup-blur-kernel",
    "--guidedmixup-condition",
    "--lr",
    "--learning-rate",
    "--lr-decay-epochs",
    "--lr-schedule",
    "--max-soft",
    "--max-eval-steps",
    "--max-train-steps",
    "--max-val-steps",
    "--method",
    "--milestones",
    "--mix-prob",
    "--momentum",
    "--no-augment",
    "--output-dir",
    "--recipe",
    "--reformulate",
    "--sal-aug-recipe",
    "--saliency-source",
    "--scheduler",
    "--seed",
    "--weight-decay",
}
LAST10_EVAL_CANDIDATES = [
    ("eval_top1_error", "error"),
    ("val_top1_error", "error"),
    ("eval_top1_accuracy", "accuracy"),
    ("val_top1", "percent"),
]
RESUME_COMPATIBILITY_KEYS = (
    "dataset",
    "recipe",
    "model",
    "model_impl_version",
    "method",
    "epochs",
    "batch_size",
    "global_batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "scheduler",
    "milestones",
    "lr_decay_rate",
    "min_learning_rate",
    "alpha",
    "decay_power",
    "max_soft",
    "reformulate",
    "method_prob",
    "use_basic_augmentation",
    "aug_recipe",
    "transform_profile",
    "mixup_no_repeat",
    "fmix_no_repeat",
    "cutmix_no_repeat",
    "catchupmix_cutmix_alpha",
    "catchupmix_num_layers",
    "catchupmix_no_repeat",
    "resizemix_scope_min",
    "resizemix_scope_max",
    "resizemix_use_alpha",
    "resizemix_no_repeat",
    "saliencymix_no_repeat",
    "guidedmixup_blur_kernel",
    "guidedmixup_condition",
    "saliency_source",
    "sal_basic_aug",
    "sal_aug_recipe",
    "cross_device_shuffle",
    "validation_split",
    "eval_on_test_each_epoch",
    "final_test",
    "final_test_checkpoint",
    "run_name",
    "run_metadata_required",
)
BEST_CHECKPOINT_EVAL_COMPATIBILITY_KEYS = tuple(
    key
    for key in RESUME_COMPATIBILITY_KEYS
    if key not in {"final_test", "final_test_checkpoint", "run_metadata_required"}
)


def _maybe_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _error_from_accuracy(value: object) -> float | None:
    accuracy = _maybe_float(value)
    if accuracy is None:
        return None
    if accuracy > 1.0:
        accuracy = accuracy / 100.0
    return 1.0 - accuracy


def _error_from_percent(value: object) -> float | None:
    percent = _maybe_float(value)
    if percent is None:
        return None
    return 1.0 - percent / 100.0


def _error_field(row: dict[str, str], field: str) -> float | None:
    error = _maybe_float(row.get(field))
    if error is None:
        return None
    return error / 100.0 if error > 1.0 else error


def _row_epoch(row: dict[str, str]) -> int | None:
    value = row.get("epoch")
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def max_recorded_epoch(rows: Iterable[dict[str, str]]) -> int:
    epochs = [_row_epoch(row) for row in rows]
    return max((epoch for epoch in epochs if epoch is not None), default=0)


def _first_available_error(row: dict[str, str], candidates: Iterable[tuple[str, str]]) -> tuple[float, str] | None:
    for field, scale in candidates:
        if scale == "error":
            error = _error_field(row, field)
        elif scale == "accuracy":
            error = _error_from_accuracy(row.get(field))
        elif scale == "percent":
            error = _error_from_percent(row.get(field))
        else:
            raise ValueError(f"Unknown metric scale: {scale}")
        if error is not None:
            return error, field
    return None


def select_error_metric(rows: list[dict[str, str]], mode: str = "auto") -> tuple[float | None, str]:
    """Return top-1 error fraction and source field from metrics rows."""

    if mode not in {"auto", "final_test", "best", "eval", "last10_median"}:
        raise ValueError(f"Unsupported summary metric mode: {mode}")

    if mode == "last10_median":
        epoch_errors = {}
        for row in rows:
            if row.get("phase") not in {"", None, "train_val", "eval"}:
                continue
            epoch = _row_epoch(row)
            if epoch is None:
                continue
            selected = _first_available_error(
                row,
                LAST10_EVAL_CANDIDATES,
            )
            if selected is not None:
                epoch_errors[epoch] = (selected[0], selected[1])
        if not epoch_errors:
            return None, "missing_last10_median"
        if len(epoch_errors) < 10:
            return None, "incomplete_last10_median"
        last_epochs = sorted(epoch_errors)[-10:]
        last_errors = [epoch_errors[epoch][0] for epoch in last_epochs]
        sources = [epoch_errors[epoch][1] for epoch in last_epochs]
        source = "last10_median:" + (sources[0] if all(value == sources[0] for value in sources) else "mixed")
        return float(np.median(np.asarray(last_errors, dtype=np.float64))), source

    if mode in {"auto", "final_test"}:
        for row in reversed(rows):
            if row.get("phase") != "final_test":
                continue
            selected = _first_available_error(
                row,
                [
                    ("test_top1_error", "error"),
                    ("test_top1_accuracy", "accuracy"),
                    ("test_top1", "percent"),
                ],
            )
            if selected is not None:
                return selected
        if mode == "final_test":
            return None, "missing_final_test"

    if mode in {"auto", "best"}:
        for row in reversed(rows):
            selected = _first_available_error(
                row,
                [
                    ("best_top1_error", "error"),
                    ("best_top1", "percent"),
                ],
            )
            if selected is not None:
                return selected
        if mode == "best":
            return None, "missing_best"

    for row in reversed(rows):
        selected = _first_available_error(
            row,
            [
                ("eval_top1_error", "error"),
                ("val_top1_error", "error"),
                ("eval_top1_accuracy", "accuracy"),
                ("val_top1", "percent"),
            ],
        )
        if selected is not None:
            return selected
    return None, "missing_metric"


def has_final_test_metric(rows: list[dict[str, str]]) -> bool:
    for row in reversed(rows):
        if row.get("phase") != "final_test":
            continue
        if _first_available_error(
            row,
            [
                ("test_top1_error", "error"),
                ("test_top1_accuracy", "accuracy"),
                ("test_top1", "percent"),
            ],
        ):
            return True
    return False


def has_expected_final_test_metric(rows: Iterable[dict[str, str]], expected_epoch: int) -> bool:
    if int(expected_epoch) <= 0:
        return has_final_test_metric(list(rows))
    for row in rows:
        if row.get("phase") != "final_test":
            continue
        if _row_epoch(row) != int(expected_epoch):
            continue
        if _first_available_error(
            row,
            [
                ("test_top1_error", "error"),
                ("test_top1_accuracy", "accuracy"),
                ("test_top1", "percent"),
            ],
        ):
            return True
    return False


def final_test_checkpoint_info(rows: Iterable[dict[str, str]], expected_epoch: int = 0) -> tuple[str, str]:
    final_test_rows = [row for row in rows if row.get("phase") == "final_test"]
    if int(expected_epoch) > 0:
        matching_rows = [row for row in final_test_rows if _row_epoch(row) == int(expected_epoch)]
        if matching_rows:
            final_test_rows = matching_rows
    if not final_test_rows:
        return "", ""
    row = final_test_rows[-1]
    return str(row.get("final_test_checkpoint") or ""), str(row.get("final_test_checkpoint_source") or "")


def final_test_checkpoint_status(
    rows: Iterable[dict[str, str]],
    expected_epoch: int,
    expected_checkpoint: object,
) -> str:
    expected = str(expected_checkpoint or "last").lower()
    if expected != "best":
        return "ok"
    checkpoint, source = final_test_checkpoint_info(rows, expected_epoch)
    checkpoint = checkpoint.lower()
    source = source.strip()
    if not checkpoint:
        return "missing_final_test_checkpoint"
    if checkpoint != "best":
        return "wrong_final_test_checkpoint"
    if source == "" or source.lower() == "current":
        return "missing_final_test_checkpoint_source"
    return "ok"


def metrics_path_from_config(root: Path, config_path: Path) -> Path:
    raw_config = load_config(str(root / config_path))
    return metrics_path_from_raw_config(root, raw_config)


def metrics_path_from_raw_config(root: Path, raw_config: dict) -> Path:
    output_dir = Path(str(raw_config.get("output_dir", "./runs/fmix")))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    run_name = str(raw_config.get("run_name") or "").strip()
    run_dir = output_dir / run_name if run_name else output_dir
    output_name = str(raw_config.get("output_name") or "").strip()
    filename = f"{output_name}.csv" if output_name else "metrics.csv"
    return run_dir / filename


def checkpoint_root_from_raw_config(root: Path, raw_config: dict) -> Path:
    checkpoint_dir = raw_config.get("checkpoint_dir")
    run_name = str(raw_config.get("run_name") or "").strip()
    if checkpoint_dir:
        checkpoint_root = Path(str(checkpoint_dir))
        if run_name:
            checkpoint_root = checkpoint_root / run_name
    else:
        checkpoint_root = metrics_path_from_raw_config(root, raw_config).parent
    return checkpoint_root if checkpoint_root.is_absolute() else root / checkpoint_root


def last_checkpoint_path_from_raw_config(root: Path, raw_config: dict) -> Path:
    return checkpoint_root_from_raw_config(root, raw_config) / "last.pt"


def best_checkpoint_path_from_raw_config(root: Path, raw_config: dict) -> Path:
    return checkpoint_root_from_raw_config(root, raw_config) / "best.pt"


def _summary_default_train_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=None,
        recipe=None,
        method=None,
        data_dir=None,
        output_dir=None,
        checkpoint=None,
        download=None,
        no_augment=False,
        aug_recipe=None,
        sal_aug_recipe=None,
        mix_prob=None,
        fmix_prob=None,
        alpha=None,
        epochs=None,
        batch_size=None,
        lr=None,
        momentum=None,
        weight_decay=None,
        scheduler=None,
        milestones=None,
        decay_power=None,
        max_soft=None,
        reformulate=None,
        guidedmixup_blur_kernel=None,
        guidedmixup_condition=None,
        saliency_source=None,
        saliency_dir=None,
        saliency_path=None,
    )


def expected_resume_config(raw_config: dict) -> dict:
    return resolved_config(_summary_default_train_args(), raw_config)


def _checkpoint_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _load_checkpoint_metadata_sidecar(path: Path) -> tuple[bool, dict | None]:
    metadata_path = _checkpoint_metadata_path(path)
    if not metadata_path.exists():
        return False, None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return True, None
    return True, metadata if isinstance(metadata, dict) else None


def load_checkpoint_metadata(path: Path) -> dict:
    sidecar_exists, sidecar_metadata = _load_checkpoint_metadata_sidecar(path)
    if sidecar_exists:
        return sidecar_metadata or {}

    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(checkpoint, dict):
        return {}
    return {
        "epoch": checkpoint.get("epoch"),
        "best_acc": checkpoint.get("best_acc"),
        "best_epoch": checkpoint.get("best_epoch"),
        "config": checkpoint.get("config"),
    }


def checkpoint_epoch(path: Path) -> int:
    metadata = load_checkpoint_metadata(path)
    try:
        return int(metadata.get("epoch") or 0)
    except (TypeError, ValueError):
        return 0


def _compatible_value(actual: object, expected: object) -> bool:
    if expected is None:
        return actual in (None, "")
    if isinstance(expected, bool):
        try:
            return _as_bool(actual, "metadata boolean") is expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return list(actual) == expected if isinstance(actual, (list, tuple)) else False
    return actual == expected


def _compatible_config_value(key: str, actual: object, expected: object) -> bool:
    if key == "model":
        try:
            return canonical_model_name(str(actual)) == canonical_model_name(str(expected))
        except ValueError:
            return False
    if key == "dataset":
        try:
            return normalize_dataset_name(str(actual)) == normalize_dataset_name(str(expected))
        except ValueError:
            return False
    if key == "method":
        return normalize_method_name(actual) == normalize_method_name(expected)
    return _compatible_value(actual, expected)


def checkpoint_resume_compatible(path: Path, raw_config: dict) -> bool:
    metadata = load_checkpoint_metadata(path)
    checkpoint_config = metadata.get("config")
    if not isinstance(checkpoint_config, dict):
        return False
    expected_config = expected_resume_config(raw_config)
    for key in RESUME_COMPATIBILITY_KEYS:
        if key not in checkpoint_config:
            return False
        if not _compatible_config_value(key, checkpoint_config.get(key), expected_config.get(key)):
            return False
    return True


def compatible_eval_only_best_checkpoint_path(path: Path, raw_config: dict) -> Path | None:
    if not path.exists():
        return None
    sidecar_exists, sidecar_metadata = _load_checkpoint_metadata_sidecar(path)
    if sidecar_exists:
        if sidecar_metadata is None:
            return None
        metadata = sidecar_metadata
    else:
        metadata = load_checkpoint_metadata(path)
    checkpoint_config = metadata.get("config")
    if not isinstance(checkpoint_config, dict):
        return None
    expected_config = expected_resume_config(raw_config)
    for key in BEST_CHECKPOINT_EVAL_COMPATIBILITY_KEYS:
        if key not in checkpoint_config:
            return None
        if not _compatible_config_value(key, checkpoint_config.get(key), expected_config.get(key)):
            return None
    return path


def epoch_level_eval_metric_epochs(rows: Iterable[dict[str, str]]) -> set[int]:
    epochs = set()
    for row in rows:
        if row.get("phase") not in {"", None, "train_val", "eval"}:
            continue
        epoch = _row_epoch(row)
        if epoch is None:
            continue
        if _first_available_error(row, LAST10_EVAL_CANDIDATES) is not None:
            epochs.add(epoch)
    return epochs


def has_expected_epoch_eval_metric(rows: Iterable[dict[str, str]], expected_epoch: int) -> bool:
    if expected_epoch <= 0:
        return True
    return int(expected_epoch) in epoch_level_eval_metric_epochs(rows)


def has_expected_last10_eval_metrics(rows: Iterable[dict[str, str]], expected_epoch: int) -> bool:
    if expected_epoch <= 0:
        return False
    if expected_epoch < 10:
        return True
    needed_epochs = set(range(int(expected_epoch) - 9, int(expected_epoch) + 1))
    return needed_epochs.issubset(epoch_level_eval_metric_epochs(rows))


def checkpoint_resume_preserves_last10(path: Path, raw_config: dict, rows: Iterable[dict[str, str]]) -> bool:
    expected_epochs = int(raw_config.get("epochs") or 0)
    if expected_epochs < 10:
        return True
    loaded_epoch = checkpoint_epoch(path)
    if loaded_epoch <= 0:
        return True
    needed_epochs = set(range(expected_epochs - 9, expected_epochs + 1))
    existing_epochs = epoch_level_eval_metric_epochs(rows)
    future_epochs = set(range(loaded_epoch + 1, expected_epochs + 1))
    return needed_epochs.issubset(existing_epochs | future_epochs)


def compatible_resume_checkpoint_path(path: Path, raw_config: dict, rows: Iterable[dict[str, str]]) -> Path | None:
    if not path.exists():
        return None
    if not checkpoint_resume_compatible(path, raw_config):
        return None
    if not checkpoint_resume_preserves_last10(path, raw_config, rows):
        return None
    return path


def run_metadata_path_from_metrics(metrics_path: Path) -> Path:
    return metrics_path.parent / "config.json"


def load_run_metadata_config(path: Path) -> dict | None:
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    resolved = metadata.get("resolved")
    if isinstance(resolved, dict):
        return resolved
    config = metadata.get("config")
    if isinstance(config, dict):
        return config
    return metadata if "method" in metadata and "dataset" in metadata else None


def run_metadata_status(metrics_path: Path, raw_config: dict) -> str:
    if not bool(raw_config.get("run_metadata_required", False)):
        return "ok"
    metadata_path = run_metadata_path_from_metrics(metrics_path)
    if not metadata_path.exists():
        return "missing_run_metadata"
    run_config = load_run_metadata_config(metadata_path)
    if not isinstance(run_config, dict):
        return "invalid_run_metadata"
    expected_config = expected_resume_config(raw_config)
    for key in RESUME_COMPATIBILITY_KEYS:
        if key not in run_config:
            return "incompatible_run_config"
        if not _compatible_config_value(key, run_config.get(key), expected_config.get(key)):
            return "incompatible_run_config"
    return "ok"


def saliency_cache_path_from_raw_config(root: Path, spec: ExperimentSpec, raw_config: dict) -> Path | None:
    if str(raw_config.get("saliency_source", "")).lower() != "batch":
        return None
    saliency_path = raw_config.get("saliency_path")
    saliency_dir = raw_config.get("saliency_dir", raw_config.get("data_dir", "./data"))
    method_name = normalize_method_name(str(raw_config.get("method") or spec.method_key))
    if method_name == "guided_sr" and not saliency_path:
        candidates = [Path(default_guided_sr_saliency_path(str(raw_config.get("dataset", "tiny_imagenet")), saliency_dir))]
    else:
        candidates = saliency_path_candidates(str(raw_config.get("dataset", "tiny_imagenet")), saliency_dir, saliency_path)
    if not candidates:
        return None
    resolved_candidates = [candidate if candidate.is_absolute() else root / candidate for candidate in candidates]
    for cache_path in resolved_candidates:
        if cache_path.exists():
            return cache_path
    return resolved_candidates[0]


def _close_numpy_mmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def saliency_cache_status(path: Path | None, raw_config: dict) -> str:
    if path is None:
        return "ok"
    if not path.exists():
        return "missing_cache"

    try:
        saliency_maps = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return "invalid_cache"

    try:
        if saliency_maps.ndim not in (3, 4):
            return "invalid_cache"
        if not saliency_array_is_finite(saliency_maps):
            return "invalid_cache"
        if saliency_maps.ndim == 4 and int(saliency_maps.shape[1]) != 1 and int(saliency_maps.shape[-1]) != 1:
            return "invalid_cache"

        dataset_name = normalize_dataset_name(str(raw_config.get("dataset", "tiny_imagenet")))
        expected_examples = EXPECTED_TRAIN_EXAMPLES.get(dataset_name)
        if expected_examples is not None and int(saliency_maps.shape[0]) != int(expected_examples):
            return "incomplete_cache"
        expected_spatial = EXPECTED_SALIENCY_SPATIAL.get(dataset_name)
        if expected_spatial is not None:
            spatial_shape = tuple(int(value) for value in saliency_maps.shape[-2:])
            if saliency_maps.ndim == 4 and int(saliency_maps.shape[-1]) == 1:
                spatial_shape = tuple(int(value) for value in saliency_maps.shape[1:3])
            if spatial_shape != tuple(expected_spatial):
                return "invalid_cache"

        method_name = normalize_method_name(str(raw_config.get("method", "")))
        expected_cache_methods = EXPECTED_SALIENCY_CACHE_METHODS.get(method_name)
        if expected_cache_methods is not None:
            metadata_path = path.with_suffix(path.suffix + ".json")
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                return "invalid_cache"
            if not isinstance(metadata, dict):
                return "invalid_cache"
            try:
                builder_version = int(metadata.get("builder_version", 0))
            except (TypeError, ValueError):
                return "invalid_cache"
            if builder_version < SALIENCY_CACHE_MIN_VERSION:
                return "invalid_cache"
            if "allow_gradient_fallback" not in metadata or bool(metadata.get("allow_gradient_fallback")):
                return "invalid_cache"
            if not bool(metadata.get("raw_unit_images")) or not bool(metadata.get("minmax_normalized")):
                return "invalid_cache"
            if str(metadata.get("method", "")).lower() not in expected_cache_methods:
                return "invalid_cache"
            metadata_dtype = str(metadata.get("dtype", "")).lower()
            if metadata_dtype not in {"float16", "float32"}:
                return "invalid_cache"
            if metadata_dtype != str(saliency_maps.dtype).lower():
                return "invalid_cache"
            expected_config = expected_resume_config(raw_config)
            if method_name == "guided_sr":
                try:
                    metadata_blur_kernel = int(metadata.get("blur_kernel"))
                    expected_blur_kernel = int(expected_config.get("guidedmixup_blur_kernel"))
                except (TypeError, ValueError):
                    return "invalid_cache"
                if metadata_blur_kernel != expected_blur_kernel:
                    return "invalid_cache"
            metadata_dataset = metadata.get("dataset")
            if metadata_dataset is None or normalize_dataset_name(str(metadata_dataset)) != dataset_name:
                return "invalid_cache"
            if metadata.get("recipe") != expected_config.get("recipe"):
                return "invalid_cache"
            if metadata.get("transform_profile") != expected_config.get("transform_profile"):
                return "invalid_cache"
            try:
                metadata_image_size = int(metadata.get("image_size"))
            except (TypeError, ValueError):
                return "invalid_cache"
            if metadata_image_size != int(get_dataset_preset(dataset_name).image_size):
                return "invalid_cache"
            if metadata.get("base_transform") != "tensor_normalize_only":
                return "invalid_cache"
            for stats_key in ("mean", "std"):
                expected_stats = expected_config.get(stats_key)
                actual_stats = metadata.get(stats_key)
                if expected_stats is None or actual_stats is None:
                    return "invalid_cache"
                try:
                    expected_array = np.asarray(list(expected_stats), dtype=np.float64)
                    actual_array = np.asarray(list(actual_stats), dtype=np.float64)
                except (TypeError, ValueError):
                    return "invalid_cache"
                if expected_array.shape != actual_array.shape or not np.allclose(
                    actual_array,
                    expected_array,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    return "invalid_cache"
            if "count" not in metadata:
                return "invalid_cache"
            try:
                metadata_count = int(metadata["count"])
            except (TypeError, ValueError):
                return "invalid_cache"
            if metadata_count != int(saliency_maps.shape[0]):
                return "invalid_cache"
            metadata_shape = metadata.get("shape")
            if metadata_shape is None:
                return "invalid_cache"
            try:
                metadata_shape_tuple = tuple(int(value) for value in metadata_shape)
            except (TypeError, ValueError):
                return "invalid_cache"
            if metadata_shape_tuple != tuple(int(value) for value in saliency_maps.shape):
                return "invalid_cache"
        return "ok"
    finally:
        _close_numpy_mmap(saliency_maps)


def _tiny_imagenet_root_candidates(root: Path, data_dir: object) -> list[Path]:
    data_root = Path(str(data_dir or "./data"))
    if not data_root.is_absolute():
        data_root = root / data_root
    return [
        data_root,
        data_root / "tiny-imagenet-200",
        data_root / "TinyImageNet",
        data_root / "tiny_imagenet",
    ]


def _tiny_imagenet_download_command(data_dir: object) -> str:
    return "bash scripts/download_tiny_imagenet.sh --data-dir " + shlex.quote(str(data_dir or "./data"))


def _has_tiny_original_layout(path: Path) -> bool:
    return (
        (path / "train").exists()
        and (path / "val").exists()
        and ((path / "wnids.txt").exists() or (path / "val" / "val_annotations.txt").exists())
    )


def _has_tiny_imagefolder_layout(path: Path) -> bool:
    train_root = path / "train"
    val_root = path / "val"
    if not train_root.exists() or not val_root.exists():
        return False
    has_train_classes = any(_tiny_class_dir_has_image(child) for child in train_root.iterdir())
    has_val_classes = any(_tiny_class_dir_has_image(child) for child in val_root.iterdir())
    return has_train_classes and has_val_classes


def _is_tiny_class_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("n") and path.name[1:].isdigit()


def _tiny_class_dir_has_image(path: Path) -> bool:
    return _is_tiny_class_dir(path) and any(
        child.is_file() and child.suffix.lower() in TINY_IMAGE_EXTENSIONS
        for child in path.rglob("*")
    )


def _count_tiny_split_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file() and child.suffix.lower() in TINY_IMAGE_EXTENSIONS)


def _tiny_imagenet_count_status(path: Path, layout: str) -> tuple[str, str]:
    train_count = _count_tiny_split_images(path / "train")
    val_count = _count_tiny_split_images(path / "val")
    expected_train = int(TINY_IMAGENET_EXPECTED_COUNTS["train"])
    expected_val = int(TINY_IMAGENET_EXPECTED_COUNTS["val"])
    detail = (
        f"{layout}; train_images={train_count}/{expected_train}; "
        f"val_images={val_count}/{expected_val}"
    )
    if train_count != expected_train or val_count != expected_val:
        return "incomplete", detail
    return "ok", detail


def tiny_imagenet_data_status(root: Path, raw_config: dict) -> tuple[str, Path, str]:
    dataset_name = normalize_dataset_name(str(raw_config.get("dataset", "tiny_imagenet")))
    candidates = _tiny_imagenet_root_candidates(root, raw_config.get("data_dir", "./data"))
    if dataset_name != "tinyimagenet":
        return "skipped", candidates[0], f"dataset={dataset_name}"

    existing_candidates = [candidate for candidate in candidates if candidate.exists()]
    for candidate in candidates:
        if _has_tiny_imagefolder_layout(candidate):
            status, detail = _tiny_imagenet_count_status(candidate, "ImageFolder train/val layout")
            return status, candidate, detail
        if _has_tiny_original_layout(candidate):
            status, detail = _tiny_imagenet_count_status(candidate, "original layout")
            return status, candidate, detail

    partial_layout_candidates = [
        candidate
        for candidate in candidates
        if (candidate / "train").exists() or (candidate / "val").exists()
    ]
    if partial_layout_candidates:
        return "invalid", partial_layout_candidates[0], "missing Tiny-ImageNet train/val layout"
    if existing_candidates:
        return "invalid", existing_candidates[0], "missing Tiny-ImageNet train/val layout"
    return (
        "missing",
        candidates[1],
        "expected tiny-imagenet-200 under data_dir; run: "
        + _tiny_imagenet_download_command(raw_config.get("data_dir", "./data")),
    )


def _count_by_status(rows: Iterable[ExperimentSummary]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def apply_preflight_saliency_arg_overrides(
    raw_config: dict,
    extra_args: Iterable[str],
    storage_base_config: dict | None = None,
) -> dict:
    config = dict(raw_config)
    data_dir = train_arg_value(extra_args, "--data-dir")
    saliency_dir = train_arg_value(extra_args, "--saliency-dir")
    saliency_path = train_arg_value(extra_args, "--saliency-path")
    resolved_saliency_dir, resolved_saliency_path = resolve_saliency_storage_paths(
        storage_base_config or raw_config,
        data_dir_override=data_dir,
        saliency_dir_override=saliency_dir,
        saliency_path_override=saliency_path,
    )
    config["saliency_dir"] = resolved_saliency_dir
    if resolved_saliency_path is not None:
        config["saliency_path"] = resolved_saliency_path
    return config


def preflight_cache_path_from_row(row: ExperimentSummary, extra_args: Iterable[str]) -> Path:
    saliency_path = train_arg_value(extra_args, "--saliency-path")
    if saliency_path is not None:
        saliency_dir = train_arg_value(extra_args, "--saliency-dir")
        return Path(relocate_relative_saliency_path(saliency_path, saliency_dir) or saliency_path)
    saliency_dir = train_arg_value(extra_args, "--saliency-dir")
    if saliency_dir is not None:
        return Path(saliency_dir) / row.prerequisite_path.name
    data_dir = train_arg_value(extra_args, "--data-dir")
    if data_dir is not None:
        return Path(data_dir) / row.prerequisite_path.name
    return row.prerequisite_path


def generated_guided_sr_batch_cache_path(extra_args: Iterable[str]) -> Path:
    saliency_path = train_arg_value(extra_args, "--saliency-path")
    if saliency_path is not None:
        saliency_dir = train_arg_value(extra_args, "--saliency-dir")
        return Path(relocate_relative_saliency_path(saliency_path, saliency_dir) or saliency_path)
    saliency_dir = train_arg_value(extra_args, "--saliency-dir") or train_arg_value(extra_args, "--data-dir") or "./data"
    return Path(default_guided_sr_saliency_path("tiny_imagenet", saliency_dir))


def generated_saliency_cache_path_for_row(row: ExperimentSummary, extra_args: Iterable[str]) -> Path | None:
    if row.prerequisite_path is not None:
        return preflight_cache_path_from_row(row, extra_args)
    if row.spec.method_key == "guided_sr" and train_arg_value(extra_args, "--saliency-source") == "batch":
        return generated_guided_sr_batch_cache_path(extra_args)
    return None


def load_preflight_data_config(root: Path, rows: Iterable[ExperimentSummary], extra_args: Iterable[str]) -> dict:
    baseline_config_path = root / TINY_IMAGENET_XLA4_SPECS[0].config_path
    if baseline_config_path.exists():
        raw_config = load_config(str(baseline_config_path))
    else:
        first_row = next(iter(rows), None)
        raw_config = load_config(str(root / first_row.spec.config_path)) if first_row is not None else {}
    return apply_preflight_train_arg_overrides(raw_config, extra_args)


def load_preflight_base_raw_config(root: Path, rows: Iterable[ExperimentSummary]) -> dict:
    baseline_config_path = root / TINY_IMAGENET_XLA4_SPECS[0].config_path
    if baseline_config_path.exists():
        return load_config(str(baseline_config_path))
    first_row = next(iter(rows), None)
    return load_config(str(root / first_row.spec.config_path)) if first_row is not None else {}


def load_preflight_raw_configs(
    root: Path,
    rows: Iterable[ExperimentSummary],
    specs: Iterable[ExperimentSpec] = (),
) -> list[tuple[str, dict]]:
    row_list = list(rows)
    spec_list = list(specs)

    configs: list[tuple[str, dict]] = []
    seen_paths: set[Path] = set()
    for row in row_list:
        config_path = root / row.spec.config_path
        if config_path in seen_paths:
            continue
        seen_paths.add(config_path)
        configs.append((row.spec.method_label, load_config(str(config_path))))
    for spec in spec_list:
        config_path = root / spec.config_path
        if config_path in seen_paths:
            continue
        seen_paths.add(config_path)
        configs.append((spec.method_label, load_config(str(config_path))))
    return configs or [("baseline", load_preflight_base_raw_config(root, row_list))]


def preflight_method_key(label: str, raw_config: dict) -> str:
    method = str(raw_config.get("method") or "").strip()
    if method:
        return normalize_method_name(method)
    normalized_label = _normalized_method_filter(label)
    return METHOD_FILTER_ALIASES.get(normalized_label, normalize_method_name(label))


def preflight_train_config_validation_status(
    root: Path,
    rows: Iterable[ExperimentSummary],
    extra_args: Iterable[str] = (),
    specs: Iterable[ExperimentSpec] = (),
) -> tuple[str, str]:
    args = list(extra_args)
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            parse_train_args(["--config", "__summary_train_arg_validation__.yaml", *args])
        configs = load_preflight_raw_configs(root, rows, specs=specs)
        for label, raw_config in configs:
            method_args = train_args_for_method(args, preflight_method_key(label, raw_config))
            parsed_args = parse_train_args(["--config", "__summary_train_arg_validation__.yaml", *method_args])
            try:
                resolved_config(parsed_args, raw_config)
            except (ValueError, TypeError) as exc:
                return "invalid", f"{label}: {exc}"
    except SystemExit as exc:
        lines = [line.strip() for line in stderr.getvalue().splitlines() if line.strip()]
        detail = next((line for line in reversed(lines) if "error:" in line), None) or (
            lines[-1] if lines else f"train arg parser exited with status {exc.code}"
        )
        if "error:" in detail:
            detail = "train parser error: " + detail.split("error:", 1)[1].strip()
        return "invalid", detail
    except OSError as exc:
        return "invalid", str(exc)
    return "ok", f"resolved config ok ({len(configs)} method{'s' if len(configs) != 1 else ''})"


def require_valid_command_train_configs(
    root: Path,
    rows: Iterable[ExperimentSummary],
    extra_args: Iterable[str] = (),
    specs: Iterable[ExperimentSpec] = (),
) -> None:
    status, detail = preflight_train_config_validation_status(root, rows, extra_args, specs=specs)
    if status != "ok":
        raise TrainArgValidationError(f"invalid --train-arg: {detail}")


def _preflight_table_detail(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _expected_tpu_devices_for_preflight(require_tpu_env: bool, num_cores: object) -> int | None:
    if not bool(require_tpu_env):
        return None
    try:
        return int(num_cores)
    except (TypeError, ValueError):
        return None


def xla_env_preflight_status(
    device: str = "xla",
    require_tpu: bool = False,
    skip_opencv_check: bool = False,
    require_venv_name: str = "",
    expected_tpu_devices: int | None = None,
) -> tuple[str, str]:
    if str(device).lower() != "xla":
        return "skipped", f"device={device}; XLA environment check is not required"

    try:
        from allthemix.cli.verify_xla_env import (
            DEFAULT_TORCH_VERSION,
            DEFAULT_TORCHVISION_VERSION,
            DEFAULT_TORCH_XLA_VERSION,
            build_checks,
        )
    except Exception as exc:
        return "invalid", _preflight_table_detail(f"verifier import failed: {exc.__class__.__name__}: {exc}")

    check_args = SimpleNamespace(
        python_version="3.10",
        torch_version=DEFAULT_TORCH_VERSION,
        torchvision_version=DEFAULT_TORCHVISION_VERSION,
        torch_xla_version=DEFAULT_TORCH_XLA_VERSION,
        require_venv_name=str(require_venv_name or ""),
        skip_device_check=not bool(require_tpu or expected_tpu_devices is not None),
        skip_opencv_check=bool(skip_opencv_check),
        require_tpu=bool(require_tpu),
        expected_tpu_devices=expected_tpu_devices,
    )
    checks = build_checks(check_args)
    failed = [check.name for check in checks if not check.ok]
    parts = [f"{check.name}={'ok' if check.ok else 'fail'}" for check in checks]
    if failed:
        command = "python -m allthemix.cli.verify_xla_env"
        if require_tpu:
            command = "PJRT_DEVICE=TPU " + command + " --require-tpu"
        else:
            command += " --skip-device-check"
        if expected_tpu_devices is not None:
            command += f" --expected-tpu-devices {expected_tpu_devices}"
        if skip_opencv_check:
            command += " --skip-opencv-check"
        if require_venv_name:
            command += f" --require-venv-name {require_venv_name}"
        parts.append("failed=" + ",".join(failed))
        parts.append("run: " + command)
    return ("ok" if not failed else "invalid"), _preflight_table_detail("; ".join(parts))


def _existing_disk_usage_path(path: Path) -> Path:
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _format_disk_space_detail(label: str, path: Path, free_gib: float, total_gib: float, min_free: float) -> str:
    threshold = "disabled" if min_free <= 0 else f"{min_free:.1f} GiB"
    return f"{label}:{path.as_posix()} free={free_gib:.1f} GiB total={total_gib:.1f} GiB min={threshold}"


def disk_space_preflight_status(root: Path, min_free_gb: float = 0.0) -> tuple[str, str]:
    return disk_space_preflight_status_for_paths({"repo": root}, min_free_gb=min_free_gb)


def disk_space_preflight_status_for_paths(paths: dict[str, Path], min_free_gb: float = 0.0) -> tuple[str, str]:
    try:
        min_free = float(min_free_gb)
    except (TypeError, ValueError):
        return "invalid", f"invalid min_free_gb={min_free_gb!r}"
    if min_free < 0:
        return "invalid", f"min_free_gb must be >= 0, got {min_free:g}"

    gib = 1024.0**3
    details = []
    failures = []
    seen: set[Path] = set()
    for label, raw_path in paths.items():
        usage_path = _existing_disk_usage_path(raw_path).resolve()
        if usage_path in seen:
            continue
        seen.add(usage_path)
        try:
            usage = shutil.disk_usage(usage_path)
        except OSError as exc:
            return "invalid", f"{label}:disk usage check failed: {exc.__class__.__name__}: {exc}"
        free_gib = float(usage.free) / gib
        total_gib = float(usage.total) / gib
        detail = _format_disk_space_detail(label, raw_path, free_gib, total_gib, min_free)
        details.append(detail)
        if min_free > 0 and free_gib < min_free:
            failures.append(label)
    if failures:
        details.append("below_min=" + ",".join(failures))
        return "invalid", "; ".join(details)
    return "ok", "; ".join(details)


def collect_preflight_disk_paths(
    root: Path,
    rows: Iterable[ExperimentSummary],
    extra_args: Iterable[str] = (),
    specs: Iterable[ExperimentSpec] = (),
) -> dict[str, Path]:
    paths: dict[str, Path] = {"repo": root}
    config_pairs = load_preflight_raw_configs(root, rows, specs=specs)
    for label, raw_config in config_pairs:
        method_args = train_args_for_method(extra_args, preflight_method_key(label, raw_config))
        config = apply_preflight_train_arg_overrides(raw_config, method_args)
        output_path = metrics_path_from_raw_config(root, config).parent
        checkpoint_path = checkpoint_root_from_raw_config(root, config)
        data_dir = Path(str(config.get("data_dir", "./data")))
        paths[f"{label}_outputs"] = output_path
        paths[f"{label}_checkpoints"] = checkpoint_path
        paths[f"{label}_data"] = data_dir if data_dir.is_absolute() else root / data_dir
        saliency_config = apply_preflight_saliency_arg_overrides(config, method_args, raw_config)
        cache_path = saliency_cache_path_from_raw_config(root, TINY_IMAGENET_XLA4_SPECS[0], saliency_config)
        if cache_path is not None:
            paths[f"{label}_saliency_cache"] = cache_path.parent
    return paths


def _resolve_preflight_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _unique_labeled_paths(paths: dict[str, Path]) -> list[tuple[str, Path]]:
    unique: dict[Path, list[str]] = {}
    order: list[Path] = []
    for label, path in paths.items():
        normalized = path.resolve(strict=False)
        if normalized not in unique:
            unique[normalized] = []
            order.append(normalized)
        unique[normalized].append(label)
    return [("+".join(unique[path]), path) for path in order]


def collect_preflight_storage_roots(
    root: Path,
    rows: Iterable[ExperimentSummary],
    extra_args: Iterable[str] = (),
    specs: Iterable[ExperimentSpec] = (),
) -> dict[str, Path]:
    paths: dict[str, Path] = {"repo": root}
    config_pairs = load_preflight_raw_configs(root, rows, specs=specs)
    for label, raw_config in config_pairs:
        method_args = train_args_for_method(extra_args, preflight_method_key(label, raw_config))
        config = apply_preflight_train_arg_overrides(raw_config, method_args)
        paths[f"{label}_output_dir"] = _resolve_preflight_path(root, config.get("output_dir", "./runs/fmix"))
        checkpoint_dir = config.get("checkpoint_dir")
        if checkpoint_dir:
            paths[f"{label}_checkpoint_dir"] = _resolve_preflight_path(root, checkpoint_dir)
        else:
            paths[f"{label}_checkpoint_dir"] = metrics_path_from_raw_config(root, config).parent
        paths[f"{label}_data_dir"] = _resolve_preflight_path(root, config.get("data_dir", "./data"))

        saliency_config = apply_preflight_saliency_arg_overrides(config, method_args, raw_config)
        if str(saliency_config.get("saliency_source", "")).lower() == "batch":
            cache_path = saliency_cache_path_from_raw_config(root, TINY_IMAGENET_XLA4_SPECS[0], saliency_config)
            if cache_path is not None:
                paths[f"{label}_saliency_dir"] = cache_path.parent
    return paths


def _path_is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _storage_root_requires_existing(label: str, path: Path, root: Path) -> bool:
    labels = label.split("+")
    if any(item.endswith("_data_dir") or item.endswith("_saliency_dir") for item in labels):
        return True
    if any(item.endswith("_output_dir") or item.endswith("_checkpoint_dir") for item in labels):
        return not _path_is_under_root(path, root)
    return True


def storage_roots_preflight_status(
    root: Path,
    rows: Iterable[ExperimentSummary],
    extra_args: Iterable[str] = (),
    specs: Iterable[ExperimentSpec] = (),
    require_existing: bool = False,
) -> tuple[str, str]:
    paths = collect_preflight_storage_roots(root, rows, extra_args=extra_args, specs=specs)
    missing_required = []
    missing_creatable = []
    for label, path in _unique_labeled_paths(paths):
        if path.exists():
            continue
        entry = f"{label}:{path.as_posix()}"
        if require_existing and _storage_root_requires_existing(label, path, root):
            missing_required.append(entry)
        else:
            missing_creatable.append(entry)
    if not missing_required and not missing_creatable:
        return "ok", "all storage roots exist"
    parts = []
    if missing_required:
        parts.append("missing_required=" + "; ".join(missing_required))
    if missing_creatable:
        parts.append("missing_creatable=" + "; ".join(missing_creatable))
    detail = "; ".join(parts)
    if missing_required:
        return "invalid", detail
    return "ok", detail + "; creatable roots may be created by training/cache commands"


def _saliency_cache_preflight(
    root: Path,
    rows: Iterable[ExperimentSummary],
    extra_args: Iterable[str] = (),
) -> tuple[str, str]:
    if train_args_use_non_batch_saliency(extra_args):
        return "skipped", "non-batch saliency source override; cache build is not required"

    parts = []
    actions = []
    eval_only_requested = train_args_have_eval_only(extra_args)
    user_checkpoint = train_args_have_checkpoint(extra_args)
    cache_extra_args = train_args_without_eval_only(extra_args) if eval_only_requested else list(extra_args)
    for row in rows:
        row_cache_args = train_args_for_method(cache_extra_args, row.spec.method_key)
        if eval_only_requested and (row.best_checkpoint_path is not None or user_checkpoint):
            continue
        fallback_path = generated_saliency_cache_path_for_row(row, row_cache_args)
        if fallback_path is None:
            continue
        try:
            base_config = load_config(str(root / row.spec.config_path))
            raw_config = apply_preflight_train_arg_overrides(base_config, row_cache_args)
            raw_config = apply_preflight_saliency_arg_overrides(raw_config, row_cache_args, base_config)
            prerequisite_path = saliency_cache_path_from_raw_config(root, row.spec, raw_config) or preflight_cache_path_from_row(
                row,
                row_cache_args,
            )
            status = saliency_cache_status(prerequisite_path, raw_config)
        except (OSError, ValueError):
            prerequisite_path = fallback_path
            status = row.prerequisite_status
        action = "ok" if status == "ok" else "will_build" if status == "missing_cache" else "will_rebuild"
        actions.append(action)
        parts.append(f"{row.spec.method_label}:{action}:{prerequisite_path.as_posix()}")
    if not parts and eval_only_requested:
        if user_checkpoint:
            return "skipped", "eval-only rows with user checkpoint do not require train saliency caches"
        return "skipped", "eval-only rows with best checkpoints do not require train saliency caches"
    if not parts:
        return "ok", "no batch-saliency caches required"
    if all(action == "ok" for action in actions):
        status = "ok"
    elif any(action == "will_rebuild" for action in actions):
        status = "will_rebuild"
    else:
        status = "will_build"
    return status, "; ".join(parts)


def _eval_only_checkpoint_preflight(
    rows: Iterable[ExperimentSummary],
    extra_args: Iterable[str] = (),
) -> tuple[str, str]:
    if not train_args_have_eval_only(extra_args):
        return "skipped", "eval-only not requested"

    pending_rows = [row for row in rows if row.status != "ok"]
    if not pending_rows:
        return "ok", "no incomplete runs need eval-only refresh"

    if train_args_have_checkpoint(extra_args):
        labels = ",".join(row.spec.method_label for row in pending_rows)
        return "ok", f"eval_only={len(pending_rows)}; full_train=0; checkpoint=user-provided; rows={labels}"

    eval_only_rows = [row for row in pending_rows if row.best_checkpoint_path is not None]
    full_train_rows = [row for row in pending_rows if row.best_checkpoint_path is None]
    if not full_train_rows:
        status = "ok"
    elif eval_only_rows:
        status = "mixed"
    else:
        status = "will_train"

    detail = f"eval_only={len(eval_only_rows)}; full_train={len(full_train_rows)}"
    if eval_only_rows:
        detail += "; best=" + ",".join(row.spec.method_label for row in eval_only_rows)
    if full_train_rows:
        detail += "; missing_best=" + ",".join(row.spec.method_label for row in full_train_rows)
    return status, detail


def render_preflight(
    root: Path,
    rows: list[ExperimentSummary],
    issues: list[ProtocolIssue],
    extra_args: Iterable[str] = (),
    device: str = "xla",
    num_cores: int = 4,
    num_workers: int = 0,
    specs: Iterable[ExperimentSpec] = (),
    check_env: bool = False,
    require_tpu_env: bool = False,
    skip_opencv_check: bool = False,
    require_venv_name: str = "",
    min_free_gb: float = 0.0,
    require_existing_storage_roots: bool = False,
) -> str:
    rows = list(rows)
    first_config = load_preflight_data_config(root, rows, extra_args)
    data_status, data_path, data_detail = tiny_imagenet_data_status(root, first_config)
    protocol_status = "ok" if not issues else f"issues={len(issues)}"
    run_status = _count_by_status(rows)
    train_arg_status, train_arg_detail = train_args_validation_status(extra_args)
    if train_arg_status == "ok":
        config_arg_status, config_arg_detail = preflight_train_config_validation_status(
            root,
            rows,
            extra_args,
            specs=specs,
        )
        if config_arg_status != "ok":
            train_arg_status = config_arg_status
            train_arg_detail = config_arg_detail
        else:
            train_arg_detail = f"{train_arg_detail}; {config_arg_detail}"
    launch_status, launch_detail = generated_launch_validation_status(device, num_cores, num_workers)
    disk_paths = collect_preflight_disk_paths(root, rows, extra_args=extra_args, specs=specs)
    disk_status, disk_detail = disk_space_preflight_status_for_paths(disk_paths, min_free_gb=min_free_gb)
    storage_status, storage_detail = storage_roots_preflight_status(
        root,
        rows,
        extra_args=extra_args,
        specs=specs,
        require_existing=bool(require_existing_storage_roots),
    )
    cache_status, cache_detail = _saliency_cache_preflight(root, rows, extra_args=extra_args)
    checkpoint_status, checkpoint_detail = _eval_only_checkpoint_preflight(rows, extra_args=extra_args)
    should_check_env = bool(check_env or require_tpu_env)
    effective_skip_opencv_check = bool(skip_opencv_check) or train_args_use_non_batch_saliency(extra_args)
    env_status, env_detail = (
        xla_env_preflight_status(
            device=device,
            require_tpu=bool(require_tpu_env),
            skip_opencv_check=effective_skip_opencv_check,
            require_venv_name=str(require_venv_name or ""),
            expected_tpu_devices=_expected_tpu_devices_for_preflight(require_tpu_env, num_cores),
        )
        if should_check_env
        else ("skipped", "pass --check-env to verify Python, torch_xla, opencv, and optional TPU visibility")
    )
    ready = (
        "ok"
        if (
            protocol_status == "ok"
            and data_status == "ok"
            and train_arg_status == "ok"
            and launch_status == "ok"
            and disk_status == "ok"
            and storage_status == "ok"
            and env_status in {"ok", "skipped"}
        )
        else "blocked"
    )
    ready_detail = (
        f"protocol={protocol_status}; data={data_status}; train_args={train_arg_status}; "
        f"launch_args={launch_status}"
    )
    if should_check_env:
        ready_detail += f"; xla_env={env_status}"
    try:
        min_free_enabled = float(min_free_gb or 0.0) > 0
    except (TypeError, ValueError):
        min_free_enabled = True
    if min_free_enabled:
        ready_detail += f"; disk_space={disk_status}"
    if bool(require_existing_storage_roots):
        ready_detail += f"; storage_roots={storage_status}"
    lines = [
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
        f"| ready_to_launch | {ready} | {ready_detail} |",
        f"| protocol | {protocol_status} | {TINY_IMAGENET_XLA4_PROTOCOL_ID} |",
        f"| launch_args | {launch_status} | {launch_detail} |",
        f"| disk_space | {disk_status} | {disk_detail} |",
        f"| storage_roots | {storage_status} | {storage_detail} |",
        f"| train_args | {train_arg_status} | {train_arg_detail} |",
        f"| tiny_imagenet_data | {data_status} | {data_path.as_posix()} ({data_detail}) |",
        f"| run_outputs | {run_status} | complete runs are skipped by generated commands unless --include-complete is used |",
        f"| best_checkpoints | {checkpoint_status} | {checkpoint_detail} |",
        f"| saliency_caches | {cache_status} | {cache_detail} |",
    ]
    if should_check_env:
        lines.insert(8, f"| xla_env | {env_status} | {env_detail} |")
    return "\n".join(lines)


def preflight_has_blockers(
    root: Path,
    issues: list[ProtocolIssue],
    rows: Iterable[ExperimentSummary] = (),
    extra_args: Iterable[str] = (),
    device: str = "xla",
    num_cores: int = 4,
    num_workers: int = 0,
    specs: Iterable[ExperimentSpec] = (),
    check_env: bool = False,
    require_tpu_env: bool = False,
    skip_opencv_check: bool = False,
    require_venv_name: str = "",
    min_free_gb: float = 0.0,
    require_existing_storage_roots: bool = False,
) -> bool:
    if issues:
        return True
    row_list = list(rows)
    if extra_args == () and row_list and all(isinstance(item, str) for item in row_list):
        extra_args = [str(item) for item in row_list]
        row_list = []
    launch_arg_status, _ = generated_launch_validation_status(device, num_cores, num_workers)
    if launch_arg_status != "ok":
        return True
    disk_paths = collect_preflight_disk_paths(root, row_list, extra_args=extra_args, specs=specs)
    disk_status, _ = disk_space_preflight_status_for_paths(disk_paths, min_free_gb=min_free_gb)
    if disk_status != "ok":
        return True
    storage_status, _ = storage_roots_preflight_status(
        root,
        row_list,
        extra_args=extra_args,
        specs=specs,
        require_existing=bool(require_existing_storage_roots),
    )
    if storage_status != "ok":
        return True
    if bool(check_env or require_tpu_env):
        effective_skip_opencv_check = bool(skip_opencv_check) or train_args_use_non_batch_saliency(extra_args)
        env_status, _ = xla_env_preflight_status(
            device=device,
            require_tpu=bool(require_tpu_env),
            skip_opencv_check=effective_skip_opencv_check,
            require_venv_name=str(require_venv_name or ""),
            expected_tpu_devices=_expected_tpu_devices_for_preflight(require_tpu_env, num_cores),
        )
        if env_status not in {"ok", "skipped"}:
            return True
    train_arg_status, _ = train_args_validation_status(extra_args)
    if train_arg_status != "ok":
        return True
    config_arg_status, _ = preflight_train_config_validation_status(root, row_list, extra_args, specs=specs)
    if config_arg_status != "ok":
        return True
    first_config = load_preflight_data_config(root, [], extra_args)
    data_status, _, _ = tiny_imagenet_data_status(root, first_config)
    return data_status != "ok"


def summary_auxiliary_config(raw_config: dict, spec: ExperimentSpec, extra_args: Iterable[str] = ()) -> dict:
    method_args = train_args_for_method(extra_args, spec.method_key)
    if not method_args:
        return raw_config
    data_dir = train_arg_value(method_args, "--data-dir")
    saliency_dir = train_arg_value(method_args, "--saliency-dir")
    saliency_path = train_arg_value(method_args, "--saliency-path")
    if data_dir is None and saliency_dir is None and saliency_path is None:
        return raw_config

    config = dict(raw_config)
    if data_dir is not None:
        config["data_dir"] = data_dir
    resolved_saliency_dir, resolved_saliency_path = resolve_saliency_storage_paths(
        raw_config,
        data_dir_override=data_dir,
        saliency_dir_override=saliency_dir,
        saliency_path_override=saliency_path,
    )
    config["saliency_dir"] = resolved_saliency_dir
    if resolved_saliency_path is not None:
        config["saliency_path"] = resolved_saliency_path
    return config


def summarize_experiment(
    root: Path,
    spec: ExperimentSpec,
    metric_mode: str = "auto",
    extra_args: Iterable[str] = (),
) -> ExperimentSummary:
    raw_config = load_config(str(root / spec.config_path))
    auxiliary_config = summary_auxiliary_config(raw_config, spec, extra_args)
    resolved = expected_resume_config(raw_config)
    metrics_path = metrics_path_from_raw_config(root, raw_config)
    resume_checkpoint_path = last_checkpoint_path_from_raw_config(root, raw_config)
    best_checkpoint_path = best_checkpoint_path_from_raw_config(root, raw_config)
    best_checkpoint_path = compatible_eval_only_best_checkpoint_path(best_checkpoint_path, raw_config)
    prerequisite_path = saliency_cache_path_from_raw_config(root, spec, auxiliary_config)
    prerequisite_status = saliency_cache_status(prerequisite_path, auxiliary_config)
    if not metrics_path.exists():
        resume_checkpoint_path = compatible_resume_checkpoint_path(resume_checkpoint_path, raw_config, [])
        return ExperimentSummary(
            spec,
            metrics_path,
            None,
            "missing_file",
            "missing",
            prerequisite_status,
            prerequisite_path,
            resume_checkpoint_path,
            best_checkpoint_path,
        )

    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    resume_checkpoint_path = compatible_resume_checkpoint_path(resume_checkpoint_path, raw_config, rows)
    if not rows:
        return ExperimentSummary(
            spec,
            metrics_path,
            None,
            "empty_file",
            "missing",
            prerequisite_status,
            prerequisite_path,
            resume_checkpoint_path,
            best_checkpoint_path,
        )

    error, source = select_error_metric(rows, mode=metric_mode)
    status = "ok" if error is not None else "missing"
    expected_epochs = int(raw_config.get("epochs") or 0)
    final_test_checkpoint, final_test_checkpoint_source = final_test_checkpoint_info(rows, expected_epochs)
    if error is not None and expected_epochs > 0 and max_recorded_epoch(rows) < expected_epochs:
        status = "incomplete"
    if (
        status == "ok"
        and bool(resolved.get("final_test", False))
        and metric_mode in {"auto", "final_test"}
        and not has_expected_final_test_metric(rows, expected_epochs)
    ):
        status = "missing_final_test"
    if (
        status == "ok"
        and bool(resolved.get("final_test", False))
        and metric_mode in {"auto", "final_test"}
    ):
        checkpoint_status = final_test_checkpoint_status(
            rows,
            expected_epochs,
            resolved.get("final_test_checkpoint"),
        )
        if checkpoint_status != "ok":
            status = checkpoint_status
    if status == "ok" and prerequisite_status != "ok":
        status = prerequisite_status
    if status == "ok":
        metadata_status = run_metadata_status(metrics_path, raw_config)
        if metadata_status != "ok":
            status = metadata_status
    if status == "ok" and not has_expected_epoch_eval_metric(rows, expected_epochs):
        status = "incomplete"
    if status == "ok" and metric_mode == "last10_median" and not has_expected_last10_eval_metrics(rows, expected_epochs):
        status = "incomplete"
    return ExperimentSummary(
        spec,
        metrics_path,
        error,
        source,
        status,
        prerequisite_status,
        prerequisite_path,
        resume_checkpoint_path,
        best_checkpoint_path,
        final_test_checkpoint,
        final_test_checkpoint_source,
    )


def summarize_experiments(
    root: Path,
    specs: Iterable[ExperimentSpec],
    metric_mode: str = "auto",
    extra_args: Iterable[str] = (),
) -> list[ExperimentSummary]:
    return [summarize_experiment(root, spec, metric_mode=metric_mode, extra_args=extra_args) for spec in specs]


def _values_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def _append_field_issue(
    issues: list[ProtocolIssue],
    spec: ExperimentSpec,
    raw_config: dict,
    field: str,
    expected: object,
) -> None:
    actual = raw_config.get(field)
    if not _values_equal(actual, expected):
        issues.append(ProtocolIssue(spec.method_key, spec.config_path, field, expected, actual))


def validate_tiny_xla4_protocol(root: Path, specs: Iterable[ExperimentSpec] = TINY_IMAGENET_XLA4_SPECS) -> list[ProtocolIssue]:
    issues: list[ProtocolIssue] = []
    for spec in specs:
        config_path = root / spec.config_path
        if not config_path.exists():
            issues.append(ProtocolIssue(spec.method_key, spec.config_path, "config_exists", True, False))
            continue
        script_path = root / script_path_for_spec(spec)
        if not script_path.exists():
            issues.append(ProtocolIssue(spec.method_key, spec.config_path, "script_exists", True, False))
        else:
            script_text = script_path.read_text()
            if "python -m allthemix.cli.train" not in script_text:
                issues.append(ProtocolIssue(spec.method_key, spec.config_path, "script_train_entrypoint", True, False))
            if 'source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"' not in script_text:
                issues.append(ProtocolIssue(spec.method_key, spec.config_path, "script_tpu_env_guard", True, False))
            if 'export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"' not in script_text:
                issues.append(ProtocolIssue(spec.method_key, spec.config_path, "script_pjrt_device_default", "TPU", "missing"))
            if spec.config_path.as_posix() not in script_text:
                issues.append(
                    ProtocolIssue(
                        spec.method_key,
                        spec.config_path,
                        "script_config",
                        spec.config_path.as_posix(),
                        "missing",
                    )
                )

        raw_config = load_config(str(config_path))
        for field, expected in COMMON_TINY_XLA4_EXPECTED.items():
            _append_field_issue(issues, spec, raw_config, field, expected)
        for field, expected in METHOD_TINY_XLA4_EXPECTED[spec.method_key].items():
            actual = normalize_method_name(raw_config.get(field, "")) if field == "method" else raw_config.get(field)
            if not _values_equal(actual, expected):
                issues.append(ProtocolIssue(spec.method_key, spec.config_path, field, expected, actual))

        resolved = expected_resume_config(raw_config)
        for field, expected in COMMON_TINY_XLA4_RESOLVED_EXPECTED.items():
            actual = resolved.get(field)
            if not _values_equal(actual, expected):
                issues.append(ProtocolIssue(spec.method_key, spec.config_path, f"resolved.{field}", expected, actual))
        for field, expected in METHOD_TINY_XLA4_RESOLVED_EXPECTED[spec.method_key].items():
            actual = resolved.get(field)
            if not _values_equal(actual, expected):
                issues.append(ProtocolIssue(spec.method_key, spec.config_path, f"resolved.{field}", expected, actual))

        expected_run_name = f"tiny_imagenet_preact_resnet18_{spec.method_key}_xla4"
        _append_field_issue(issues, spec, raw_config, "run_name", expected_run_name)

        if spec.method_key == "fmix":
            section = raw_config.get("fmix", {})
            for field, expected in FMIX_SECTION_EXPECTED.items():
                actual = section.get(field) if isinstance(section, dict) else None
                if not _values_equal(actual, expected):
                    issues.append(ProtocolIssue(spec.method_key, spec.config_path, f"fmix.{field}", expected, actual))
        if spec.method_key == "mixup":
            section = raw_config.get("mixup", {})
            for field, expected in MIXUP_SECTION_EXPECTED.items():
                actual = section.get(field) if isinstance(section, dict) else None
                if not _values_equal(actual, expected):
                    issues.append(ProtocolIssue(spec.method_key, spec.config_path, f"mixup.{field}", expected, actual))
        if spec.method_key == "saliencymix":
            cache_script_path = root / "scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh"
            if not cache_script_path.exists():
                issues.append(ProtocolIssue(spec.method_key, spec.config_path, "cache_script_exists", True, False))
            else:
                cache_script = cache_script_path.read_text()
                if 'source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"' not in cache_script:
                    issues.append(ProtocolIssue(spec.method_key, spec.config_path, "cache_script_tpu_env_guard", True, False))
                if spec.config_path.as_posix() not in cache_script:
                    issues.append(
                        ProtocolIssue(
                            spec.method_key,
                            spec.config_path,
                            "cache_script_config",
                            spec.config_path.as_posix(),
                            "missing",
                        )
                    )
                if "--allow-gradient-fallback" in cache_script or "--method gradient" in cache_script:
                    issues.append(
                        ProtocolIssue(
                            spec.method_key,
                            spec.config_path,
                            "cache_script_strict_opencv",
                            "no gradient fallback",
                            "fallback enabled",
                        )
                    )

    return issues


def format_error_percent(error: float | None) -> str:
    return "--" if error is None else f"{100.0 * error:.2f}"


def format_complete_error_percent(row: ExperimentSummary) -> str:
    return format_error_percent(row.error) if row.status == "ok" else "--"


def complete_tiny_best_method_keys(rows: Iterable[ExperimentSummary]) -> set[str]:
    complete_rows = [row for row in rows if row.status == "ok" and row.error is not None]
    if not complete_rows:
        return set()
    best_error = min(float(row.error) for row in complete_rows)
    return {row.spec.method_key for row in complete_rows if np.isclose(float(row.error), best_error)}


def format_complete_error_percent_for_table(row: ExperimentSummary, best_method_keys: set[str]) -> str:
    value = format_complete_error_percent(row)
    if value != "--" and row.spec.method_key in best_method_keys:
        return rf"\textbf{{{value}}}"
    return value


def script_path_for_spec(spec: ExperimentSpec) -> Path:
    if spec.script_path is not None:
        return spec.script_path
    return Path(f"scripts/experiment_run/run_tiny_imagenet_preact_resnet18_{spec.method_key}_xla4.sh")


def training_command(
    spec: ExperimentSpec,
    device: str = "xla",
    num_cores: int = 4,
    num_workers: int = 0,
    extra_args: Iterable[str] = (),
    checkpoint: str | Path | None = None,
) -> str:
    parts = [
        "bash",
        script_path_for_spec(spec).as_posix(),
        "--device",
        device,
        "--num-cores",
        str(num_cores),
        "--num-workers",
        str(num_workers),
    ]
    if checkpoint is not None:
        checkpoint_arg = checkpoint.as_posix() if isinstance(checkpoint, Path) else str(checkpoint)
        parts.extend(["--checkpoint", checkpoint_arg])
    parts.extend(extra_args)
    return " ".join(shlex.quote(part) for part in parts)


def train_arg_applies_to_method(flag: str, method_key: str) -> bool:
    allowed_methods = METHOD_SPECIFIC_TRAIN_VALUE_ARGS.get(flag) or METHOD_SPECIFIC_TRAIN_FLAG_ARGS.get(flag)
    if allowed_methods is None:
        return True
    return normalize_method_name(method_key) in allowed_methods


def train_args_for_method(extra_args: Iterable[str], method_key: str) -> list[str]:
    args = list(extra_args)
    filtered: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        flag = arg.split("=", 1)[0]
        if not train_arg_applies_to_method(flag, method_key):
            if flag in METHOD_SPECIFIC_TRAIN_VALUE_ARGS and "=" not in arg and index + 1 < len(args):
                index += 2
            else:
                index += 1
            continue
        filtered.append(arg)
        if "=" not in arg and flag in METHOD_SPECIFIC_TRAIN_VALUE_ARGS and index + 1 < len(args):
            filtered.append(args[index + 1])
            index += 2
            continue
        index += 1
    return filtered


def cache_arg_flag_from_train_arg(flag: str, method_key: str | None = None) -> str | None:
    if flag in CACHE_COMMAND_FORWARD_VALUE_ARGS:
        return flag
    if flag not in CACHE_COMMAND_RENAMED_VALUE_ARGS:
        return None
    normalized_method = normalize_method_name(method_key) if method_key is not None else None
    if flag == "--guidedmixup-blur-kernel" and normalized_method not in {None, "guided_sr"}:
        return None
    return CACHE_COMMAND_RENAMED_VALUE_ARGS[flag]


def cache_args_from_train_args(extra_args: Iterable[str], method_key: str | None = None) -> list[str]:
    args = list(extra_args)
    saliency_dir = train_arg_value(args, "--saliency-dir")
    cache_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if "=" in arg:
            flag, value = arg.split("=", 1)
            cache_flag = cache_arg_flag_from_train_arg(flag, method_key)
            if cache_flag is not None:
                if flag == "--saliency-path":
                    value = relocate_relative_saliency_path(value, saliency_dir) or value
                cache_args.extend([cache_flag, value])
            index += 1
            continue

        cache_flag = cache_arg_flag_from_train_arg(arg, method_key)
        if cache_flag is not None:
            if index + 1 < len(args):
                value = args[index + 1]
                if arg == "--saliency-path":
                    value = relocate_relative_saliency_path(value, saliency_dir) or value
                cache_args.append(cache_flag)
                cache_args.append(value)
                index += 2
                continue
        index += 1
    return cache_args


def saliency_cache_command(
    spec: ExperimentSpec,
    overwrite: bool = False,
    extra_args: Iterable[str] = (),
    num_workers: int = 0,
) -> str:
    script = (
        "scripts/experiment_run/build_tiny_imagenet_guided_sr_cache.sh"
        if spec.method_key == "guided_sr"
        else "scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh"
    )
    parts = [
        "bash",
        script,
    ]
    if overwrite:
        parts.append("--overwrite")
    parts.extend(["--num-workers", str(int(num_workers))])
    parts.extend(cache_args_from_train_args(extra_args, method_key=spec.method_key))
    if (
        spec.method_key == "guided_sr"
        and train_arg_value(extra_args, "--saliency-source") == "batch"
        and train_arg_value(extra_args, "--saliency-path") is None
    ):
        saliency_dir = train_arg_value(extra_args, "--saliency-dir") or train_arg_value(extra_args, "--data-dir") or "./data"
        parts.extend(["--output", default_guided_sr_saliency_path("tiny_imagenet", saliency_dir)])
    return " ".join(shlex.quote(part) for part in parts)


def train_args_use_non_batch_saliency(extra_args: Iterable[str]) -> bool:
    saliency_source = train_arg_value(extra_args, "--saliency-source")
    if saliency_source is None:
        return False
    return saliency_source.lower() != "batch"


def train_args_have_checkpoint(extra_args: Iterable[str]) -> bool:
    for arg in extra_args:
        if arg in {"--checkpoint", "--resume-checkpoint"}:
            return True
        if arg.startswith("--checkpoint=") or arg.startswith("--resume-checkpoint="):
            return True
    return False


def train_args_have_eval_only(extra_args: Iterable[str]) -> bool:
    for arg in extra_args:
        if arg.split("=", 1)[0] == "--eval-only":
            return True
    return False


def train_args_without_eval_only(extra_args: Iterable[str]) -> list[str]:
    return [arg for arg in extra_args if arg.split("=", 1)[0] != "--eval-only"]


def train_arg_value(extra_args: Iterable[str], *flags: str) -> str | None:
    args = list(extra_args)
    flag_set = set(flags)
    value = None
    for index, arg in enumerate(args):
        if arg in flag_set and index + 1 < len(args):
            value = args[index + 1]
        if "=" in arg:
            flag, parsed_value = arg.split("=", 1)
            if flag in flag_set:
                value = parsed_value
    return value


def train_args_validation_status(extra_args: Iterable[str]) -> tuple[str, str]:
    args = list(extra_args)
    if not args:
        return "ok", "no extra train args"

    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in RESERVED_GENERATED_COMMAND_TRAIN_ARGS:
            hint = RESERVED_GENERATED_COMMAND_TRAIN_ARG_HINTS[flag]
            return (
                "invalid",
                f"{flag} is reserved by generated commands; {hint}",
            )

    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            parse_train_args(["--config", "__summary_train_arg_validation__.yaml", *args])
    except SystemExit as exc:
        lines = [line.strip() for line in stderr.getvalue().splitlines() if line.strip()]
        detail = next((line for line in reversed(lines) if "error:" in line), None)
        if detail is None and lines:
            detail = lines[-1]
        if detail is None:
            detail = f"train arg parser exited with status {exc.code}"
        elif "error:" in detail:
            detail = "train parser error: " + detail.split("error:", 1)[1].strip()
        return "invalid", detail
    return "ok", "extra_args=" + shlex.join(args)


def require_valid_train_args(extra_args: Iterable[str]) -> None:
    status, detail = train_args_validation_status(extra_args)
    if status != "ok":
        raise TrainArgValidationError(f"invalid --train-arg: {detail}")


def generated_launch_validation_status(device: str, num_cores: int, num_workers: int) -> tuple[str, str]:
    try:
        core_count = int(num_cores)
        worker_count = int(num_workers)
    except (TypeError, ValueError):
        return "invalid", f"num_cores and num_workers must be integers; got {num_cores!r}, {num_workers!r}"

    if core_count < 1:
        return "invalid", f"num_cores must be >= 1; got {core_count}"
    if worker_count < 0:
        return "invalid", f"num_workers must be >= 0; got {worker_count}"

    expected_tpu_devices = int(TINY_IMAGENET_XLA4_PROTOCOL["expected_tpu_devices"])
    if str(device) == "xla" and core_count != expected_tpu_devices:
        return (
            "invalid",
            f"tiny-imagenet-xla4 expects --num-cores {expected_tpu_devices} with --device xla; got {core_count}",
        )
    return "ok", f"device={device}; num_cores={core_count}; num_workers={worker_count}"


def require_valid_generated_launch(device: str, num_cores: int, num_workers: int) -> None:
    status, detail = generated_launch_validation_status(device, num_cores, num_workers)
    if status != "ok":
        raise TrainArgValidationError(f"invalid generated command launch: {detail}")


def apply_preflight_train_arg_overrides(raw_config: dict, extra_args: Iterable[str]) -> dict:
    config = dict(raw_config)
    data_dir = train_arg_value(extra_args, "--data-dir")
    checkpoint_dir = train_arg_value(extra_args, "--checkpoint-dir")
    dataset = train_arg_value(extra_args, "--dataset")
    recipe = train_arg_value(extra_args, "--recipe")
    guidedmixup_blur_kernel = train_arg_value(extra_args, "--guidedmixup-blur-kernel")
    saliency_source = train_arg_value(extra_args, "--saliency-source")
    if data_dir is not None:
        config["data_dir"] = data_dir
    if checkpoint_dir is not None:
        config["checkpoint_dir"] = checkpoint_dir
    if dataset is not None:
        config["dataset"] = dataset
    if recipe is not None:
        config["recipe"] = recipe
    if guidedmixup_blur_kernel is not None:
        config["guidedmixup_blur_kernel"] = guidedmixup_blur_kernel
    if saliency_source is not None:
        config["saliency_source"] = saliency_source
    return config


def train_args_have_saliency_cache_path_override(extra_args: Iterable[str]) -> bool:
    for arg in extra_args:
        flag = arg.split("=", 1)[0]
        if flag in {"--data-dir", "--saliency-dir", "--saliency-path"}:
            return True
    return False


def train_args_have_saliency_cache_content_override(
    extra_args: Iterable[str],
    method_key: str | None = None,
) -> bool:
    for arg in extra_args:
        flag = arg.split("=", 1)[0]
        if flag == "--recipe":
            return True
        if flag == "--guidedmixup-blur-kernel" and method_key in {None, "guided_sr"}:
            return True
    return False


def train_args_disable_auto_resume(extra_args: Iterable[str]) -> bool:
    for arg in extra_args:
        flag = arg.split("=", 1)[0]
        if flag in AUTO_RESUME_UNSAFE_TRAIN_ARGS:
            return True
    return False


def render_markdown(rows: list[ExperimentSummary]) -> str:
    lines = [
        "| Type | Method | Tiny-ImageNet Top-1 Err | Candidate Err | Status | Source | Metrics |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.spec.type_name} | {row.spec.method_label} | {format_complete_error_percent(row)} | "
            f"{format_error_percent(row.error)} | {row.status} | {row.metric_source} | "
            f"{row.metrics_path.as_posix()} |"
        )
    return "\n".join(lines)


def render_csv(rows: list[ExperimentSummary]) -> str:
    fields = [
        "protocol_id",
        "type",
        "method",
        "method_key",
        "tiny_imagenet_top1_error",
        "candidate_top1_error",
        "metric_source",
        "status",
        "prerequisite_status",
        "config_path",
        "metrics_path",
        "resume_checkpoint_path",
        "best_checkpoint_path",
        "final_test_checkpoint",
        "final_test_checkpoint_source",
        "prerequisite_path",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "protocol_id": TINY_IMAGENET_XLA4_PROTOCOL_ID,
                "type": row.spec.type_name,
                "method": row.spec.method_label,
                "method_key": row.spec.method_key,
                "tiny_imagenet_top1_error": format_complete_error_percent(row),
                "candidate_top1_error": format_error_percent(row.error),
                "metric_source": row.metric_source,
                "status": row.status,
                "prerequisite_status": row.prerequisite_status,
                "config_path": row.spec.config_path.as_posix(),
                "metrics_path": row.metrics_path.as_posix(),
                "resume_checkpoint_path": (
                    row.resume_checkpoint_path.as_posix()
                    if row.resume_checkpoint_path is not None
                    else ""
                ),
                "best_checkpoint_path": (
                    row.best_checkpoint_path.as_posix()
                    if row.best_checkpoint_path is not None
                    else ""
                ),
                "final_test_checkpoint": row.final_test_checkpoint,
                "final_test_checkpoint_source": row.final_test_checkpoint_source,
                "prerequisite_path": (
                    row.prerequisite_path.as_posix()
                    if row.prerequisite_path is not None
                    else ""
                ),
            }
        )
    return buffer.getvalue().rstrip("\n")


def render_status(rows: list[ExperimentSummary]) -> str:
    lines = [
        "| Method | Status | Prereq | Tiny-ImageNet Top-1 Err | Source | Final Test Checkpoint | Metrics |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        prereq = row.prerequisite_status
        if row.prerequisite_path is not None:
            prereq = f"{prereq}:{row.prerequisite_path.as_posix()}"
        final_test_checkpoint = row.final_test_checkpoint_source or row.final_test_checkpoint
        lines.append(
            "| "
            f"{row.spec.method_label} | {row.status} | {prereq} | {format_error_percent(row.error)} | "
            f"{row.metric_source} | {final_test_checkpoint} | {row.metrics_path.as_posix()} |"
        )
    return "\n".join(lines)


def render_json(rows: list[ExperimentSummary], metric_mode: str = "auto") -> str:
    best_method_keys = complete_tiny_best_method_keys(rows)
    payload = {
        "preset": "tiny-imagenet-xla4",
        "protocol_id": TINY_IMAGENET_XLA4_PROTOCOL_ID,
        "protocol": TINY_IMAGENET_XLA4_PROTOCOL,
        "metric_mode": metric_mode,
        "rows": [],
        "best_complete_method_keys": sorted(best_method_keys),
    }
    for row in rows:
        table_value = format_complete_error_percent(row)
        candidate_value = format_error_percent(row.error)
        candidate_error = None if row.error is None else float(row.error)
        payload["rows"].append(
            {
                "type": row.spec.type_name,
                "method": row.spec.method_label,
                "method_key": row.spec.method_key,
                "tiny_imagenet_top1_error": table_value,
                "tiny_imagenet_top1_error_fraction": candidate_error if row.status == "ok" else None,
                "candidate_top1_error": candidate_value,
                "candidate_top1_error_fraction": candidate_error,
                "metric_source": row.metric_source,
                "status": row.status,
                "prerequisite_status": row.prerequisite_status,
                "metrics_path": row.metrics_path.as_posix(),
                "config_path": row.spec.config_path.as_posix(),
                "is_best_complete_tiny_imagenet": row.spec.method_key in best_method_keys,
                "resume_checkpoint_path": (
                    row.resume_checkpoint_path.as_posix()
                    if row.resume_checkpoint_path is not None
                    else None
                ),
                "best_checkpoint_path": (
                    row.best_checkpoint_path.as_posix()
                    if row.best_checkpoint_path is not None
                    else None
                ),
                "final_test_checkpoint": row.final_test_checkpoint,
                "final_test_checkpoint_source": row.final_test_checkpoint_source,
                "prerequisite_path": (
                    row.prerequisite_path.as_posix()
                    if row.prerequisite_path is not None
                    else None
                ),
            }
        )
    return json.dumps(payload, indent=2) + "\n"


def command_rows_for_generation(rows: Iterable[ExperimentSummary], include_complete: bool = False) -> list[ExperimentSummary]:
    return [row for row in rows if row.status != "ok" or include_complete]


def render_commands(
    rows: list[ExperimentSummary],
    device: str = "xla",
    num_cores: int = 4,
    num_workers: int = 0,
    include_complete: bool = False,
    extra_args: Iterable[str] = (),
) -> str:
    commands = []
    prerequisite_commands = set()
    extra_args = list(extra_args)
    require_valid_generated_launch(device, num_cores, num_workers)
    require_valid_train_args(extra_args)
    user_checkpoint = train_args_have_checkpoint(extra_args)
    eval_only_requested = train_args_have_eval_only(extra_args)
    command_rows = command_rows_for_generation(rows, include_complete=include_complete)
    if user_checkpoint and len(command_rows) > 1:
        labels = ", ".join(row.spec.method_key for row in command_rows)
        raise TrainArgValidationError(
            "manual --checkpoint can only be used with one selected method; "
            f"pass --method to select exactly one row. selected={labels}"
        )
    for row in command_rows:
        method_extra_args = train_args_for_method(extra_args, row.spec.method_key)
        method_unsafe_auto_checkpoint = train_args_disable_auto_resume(method_extra_args)
        row_uses_eval_only = eval_only_requested and (
            user_checkpoint or (row.best_checkpoint_path is not None and not method_unsafe_auto_checkpoint)
        )
        row_extra_args = method_extra_args if row_uses_eval_only else train_args_without_eval_only(method_extra_args)
        skip_saliency_cache = row_uses_eval_only or train_args_use_non_batch_saliency(row_extra_args)
        disable_auto_resume = user_checkpoint or train_args_disable_auto_resume(row_extra_args)
        effective_prerequisite_path = generated_saliency_cache_path_for_row(row, row_extra_args)
        force_saliency_cache = train_args_have_saliency_cache_path_override(
            row_extra_args
        ) or train_args_have_saliency_cache_content_override(row_extra_args, row.spec.method_key)
        force_saliency_cache_overwrite = train_args_have_saliency_cache_content_override(
            row_extra_args,
            row.spec.method_key,
        )
        needs_saliency_cache = row.prerequisite_status != "ok" or (
            force_saliency_cache and effective_prerequisite_path is not None
        ) or (
            effective_prerequisite_path is not None
            and row.prerequisite_path is None
            and train_arg_value(row_extra_args, "--saliency-source") == "batch"
        )
        if needs_saliency_cache and not skip_saliency_cache:
            command = saliency_cache_command(
                row.spec,
                overwrite=force_saliency_cache_overwrite
                or row.prerequisite_status not in {"ok", "missing_cache"},
                extra_args=row_extra_args,
                num_workers=num_workers,
            )
            if command not in prerequisite_commands:
                commands.append(command)
                prerequisite_commands.add(command)
        commands.append(
            training_command(
                row.spec,
                device=device,
                num_cores=num_cores,
                num_workers=num_workers,
                extra_args=row_extra_args,
                checkpoint=(
                    None
                    if user_checkpoint
                    else row.best_checkpoint_path
                    if row_uses_eval_only
                    else None
                    if disable_auto_resume
                    else row.resume_checkpoint_path
                ),
            )
        )
    if not commands:
        return "# All tiny-imagenet-xla4 runs already have a selected metric."
    return "\n".join(commands)


def render_latex(rows: list[ExperimentSummary]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"{row.spec.type_name} & {row.spec.method_label} & -- & -- & -- & "
            f"{format_complete_error_percent(row)} & -- & -- \\\\"
        )
    return "\n".join(lines)


def render_latex_table(rows: list[ExperimentSummary]) -> str:
    best_method_keys = complete_tiny_best_method_keys(rows)
    tiny_errors = {
        row.spec.method_key: format_complete_error_percent_for_table(row, best_method_keys)
        for row in rows
    }
    lines = [
        r"Type & Method & CIFAR-10 & CIFAR-100 & STL-10 & Tiny-ImageNet & Cars196 & CUB \\",
        r"\midrule",
    ]
    for index, (type_name, method, method_key, cifar10, cifar100, stl10, cars196, cub) in enumerate(MAIN_TABLE_ROWS):
        if index in {8, 11}:
            lines.append(r"\midrule")
        tiny = tiny_errors.get(method_key, "--") if method_key is not None else "--"
        lines.append(
            f"{type_name} & {method} & {cifar10} & {cifar100} & {stl10} & {tiny} & {cars196} & {cub} \\\\"
        )
    lines.append(r"\bottomrule")
    return "\n".join(lines)


def render_protocol(issues: list[ProtocolIssue]) -> str:
    if not issues:
        return "tiny-imagenet-xla4 protocol: ok"
    lines = [
        "| Method | Config | Field | Expected | Actual |",
        "| --- | --- | --- | --- | --- |",
    ]
    for issue in issues:
        lines.append(
            f"| {issue.method_key} | {issue.config_path.as_posix()} | {issue.field} | "
            f"{issue.expected!r} | {issue.actual!r} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize AllTheMiXLA experiment metrics.")
    parser.add_argument("--root", default=".", help="Repository root containing configs and outputs.")
    parser.add_argument("--preset", choices=["tiny-imagenet-xla4"], default="tiny-imagenet-xla4")
    parser.add_argument("--metric", choices=["auto", "final_test", "best", "eval", "last10_median"], default="auto")
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json", "latex", "latex-table", "status", "commands", "protocol", "preflight"],
        default="markdown",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xla"], default="xla", help="Device argument used by --format commands.")
    parser.add_argument("--num-cores", type=int, default=4, help="TPU core count used by --format commands.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers used by --format commands.")
    parser.add_argument("--include-complete", action="store_true", help="Include completed runs in --format commands.")
    parser.add_argument("--method", action="append", default=[], help="Limit the selected Tiny-ImageNet methods; repeat for multiple methods.")
    parser.add_argument("--train-arg", action="append", default=[], help="Extra argument token appended to each generated train command.")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="With --format preflight, also verify the Python/PyTorch/XLA/OpenCV environment.",
    )
    parser.add_argument(
        "--require-tpu-env",
        action="store_true",
        help="With --format preflight, require visible TPU devices in the XLA environment check.",
    )
    parser.add_argument(
        "--skip-opencv-check",
        action="store_true",
        help="Skip the OpenCV saliency backend in --check-env. Only use for non-SaliencyMix debug runs.",
    )
    parser.add_argument(
        "--require-venv-name",
        default="",
        help="With --format preflight and --check-env, require this Python virtualenv directory name.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help="With --format preflight, block launch when the repo filesystem has less than this many free GiB.",
    )
    parser.add_argument(
        "--require-existing-storage-roots",
        action="store_true",
        help="With --format preflight, block when configured data/output/checkpoint/saliency root directories do not exist.",
    )
    parser.add_argument("--require-complete", action="store_true", help="Exit with status 1 if any selected metric is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    try:
        specs = filter_specs_by_method(TINY_IMAGENET_XLA4_SPECS, args.method)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.format == "protocol":
        issues = validate_tiny_xla4_protocol(root, specs)
        print(render_protocol(issues))
        if bool(args.require_complete) and issues:
            raise SystemExit(1)
        return

    if args.format == "preflight":
        issues = validate_tiny_xla4_protocol(root, specs)
        rows = summarize_experiments(root, specs, metric_mode=args.metric, extra_args=args.train_arg)
        print(
            render_preflight(
                root,
                rows,
                issues,
                extra_args=args.train_arg,
                device=args.device,
                num_cores=args.num_cores,
                num_workers=args.num_workers,
                specs=specs,
                check_env=bool(args.check_env or args.require_tpu_env),
                require_tpu_env=bool(args.require_tpu_env),
                skip_opencv_check=bool(args.skip_opencv_check),
                require_venv_name=str(args.require_venv_name or ""),
                min_free_gb=float(args.min_free_gb),
                require_existing_storage_roots=bool(args.require_existing_storage_roots),
            )
        )
        if bool(args.require_complete) and preflight_has_blockers(
            root,
            issues,
            rows=rows,
            extra_args=args.train_arg,
            device=args.device,
            num_cores=args.num_cores,
            num_workers=args.num_workers,
            specs=specs,
            check_env=bool(args.check_env or args.require_tpu_env),
            require_tpu_env=bool(args.require_tpu_env),
            skip_opencv_check=bool(args.skip_opencv_check),
            require_venv_name=str(args.require_venv_name or ""),
            min_free_gb=float(args.min_free_gb),
            require_existing_storage_roots=bool(args.require_existing_storage_roots),
        ):
            raise SystemExit(1)
        return

    rows = summarize_experiments(root, specs, metric_mode=args.metric, extra_args=args.train_arg)
    if args.format == "markdown":
        print(render_markdown(rows))
    elif args.format == "csv":
        print(render_csv(rows))
    elif args.format == "json":
        print(render_json(rows, metric_mode=args.metric), end="")
    elif args.format == "latex":
        print(render_latex(rows))
    elif args.format == "latex-table":
        print(render_latex_table(rows))
    elif args.format == "status":
        print(render_status(rows))
    elif args.format == "commands":
        try:
            command_rows = command_rows_for_generation(rows, include_complete=bool(args.include_complete))
            if command_rows:
                require_valid_command_train_configs(root, command_rows, args.train_arg, specs=specs)
            command_text = render_commands(
                rows,
                device=args.device,
                num_cores=args.num_cores,
                num_workers=args.num_workers,
                include_complete=bool(args.include_complete),
                extra_args=args.train_arg,
            )
        except TrainArgValidationError as exc:
            raise SystemExit(str(exc)) from None
        print(command_text)
    else:
        raise ValueError(f"Unsupported output format: {args.format}")

    if bool(args.require_complete) and (
        any(row.status != "ok" for row in rows) or validate_tiny_xla4_protocol(root, specs)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
