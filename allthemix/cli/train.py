"""Train MixUp/FMIX PreAct-ResNet18 on CIFAR-10/100 or Tiny-ImageNet.

Use ``python -m allthemix.cli.train --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
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
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from allthemix.data import attach_train_saliency_maps, build_datasets
from allthemix.data.datasets import IMAGENET_A_INDICES_IN_1K, IMAGENET_A_NUM_CLASSES
from allthemix.methods import CatchUpMix, CutMix, FMix, GuidedSR, MixUp, ResizeMix, SaliencyMix
from allthemix.methods.guided_sr import denormalize_images_for_saliency
from allthemix.networks import build_model, canonical_model_name, model_impl_version
from allthemix.cli.presets import (
    DATASET_ALIASES,
    DATASETS,
    RECIPES,
    get_dataset_preset,
    get_recipe_preset,
    normalize_dataset_name,
    preset_dict,
)
from allthemix.training.losses import fmix_cross_entropy, mixup_cross_entropy


METHOD_ALIASES = {
    "catch_up_mix": "catchupmix",
    "catchup_mix": "catchupmix",
    "catch-up-mix": "catchupmix",
    "cut_mix": "cutmix",
    "cut-mix": "cutmix",
    "f_mix": "fmix",
    "f-mix": "fmix",
    "guided-sr": "guided_sr",
    "guided-mixup": "guided_sr",
    "guided_mixup": "guided_sr",
    "guidedmixup": "guided_sr",
    "guidedsr": "guided_sr",
    "guidedmixup_sr": "guided_sr",
    "guidedmixup-sr": "guided_sr",
    "guided_mixup_sr": "guided_sr",
    "mix_up": "mixup",
    "mix-up": "mixup",
    "resize": "resizemix",
    "resize_mix": "resizemix",
    "resize-mix": "resizemix",
    "saliency_mix": "saliencymix",
    "saliency-mix": "saliencymix",
}
METHOD_CHOICES = sorted(
    {
        "baseline",
        "catchupmix",
        "catchup_mix",
        "catch_up_mix",
        "cutmix",
        "eval",
        "fmix",
        "guided_sr",
        "mixup",
        "none",
        "resizemix",
        "saliencymix",
        *METHOD_ALIASES.keys(),
    }
)
MAX_NUMPY_SEED = 2**32 - 1


def normalize_method_name(name: Any) -> str:
    method = str(name).lower()
    return METHOD_ALIASES.get(method, method)


def _unique_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def method_section_aliases(method_name: str) -> list[str]:
    method = normalize_method_name(method_name)
    aliases = [method]
    aliases.extend(alias for alias, target in METHOD_ALIASES.items() if target == method)
    return _unique_strings(aliases)


def _method_prefixed_keys(method_name: str, suffix: str) -> list[str]:
    return [f"{prefix}_{suffix}" for prefix in method_section_aliases(method_name)]


def _method_is_active(active_method: str, target_method: str) -> bool:
    return normalize_method_name(active_method) == normalize_method_name(target_method)


def _method_specific_value(
    raw_config: dict[str, Any],
    active_method: str,
    target_method: str,
    section: dict[str, Any],
    suffix: str,
    default: Any,
    section_key: str | None = None,
    extra_keys: list[str] | tuple[str, ...] = (),
) -> Any:
    if not _method_is_active(active_method, target_method):
        return default
    key = section_key or suffix
    return _first_config_value(
        raw_config,
        _method_prefixed_keys(target_method, suffix) + list(extra_keys),
        section.get(key, default),
    )


def _optional_xla_import() -> dict[str, Any] | None:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.runtime as xr
    except ModuleNotFoundError:
        return None
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch/XLA import failed. This usually means torch and torch_xla wheels do not match; "
            "reinstall matching versions inside the active environment."
        ) from exc
    return {"xm": xm, "pl": pl, "xr": xr}


def _optional_xla_launcher():
    try:
        import torch_xla
    except ModuleNotFoundError:
        return None
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch/XLA import failed. This usually means torch and torch_xla wheels do not match; "
            "reinstall matching versions inside the active environment."
        ) from exc
    if hasattr(torch_xla, "launch"):
        def _launch(fn, args=(), start_method="spawn", debug_single_process=False):
            del start_method
            return torch_xla.launch(fn, args=args, debug_single_process=debug_single_process)

        return _launch

    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ModuleNotFoundError:
        return None
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch/XLA multiprocessing import failed. This usually means torch and torch_xla wheels do not match; "
            "reinstall matching versions inside the active environment."
        ) from exc

    def _spawn(fn, args=(), start_method="spawn", debug_single_process=False):
        if debug_single_process:
            return fn(0, *args)
        return xmp.spawn(fn, args=args, nprocs=None, start_method=start_method)

    return _spawn


def configure_xla_launch_environment(args: argparse.Namespace) -> bool:
    num_cores = int(args.num_cores)
    num_workers = int(getattr(args, "num_workers", 0))
    if num_cores < 1:
        raise ValueError(f"--num-cores must be >= 1, got {args.num_cores}.")
    if num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {getattr(args, 'num_workers', None)}.")
    should_spawn = args.device == "xla" and num_cores > 1
    if args.device == "xla":
        os.environ["TPU_NUM_DEVICES"] = str(num_cores)
    return should_spawn


def parse_seed_arg(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seed must be an integer.") from exc
    try:
        return validate_seed(seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_step_limit_arg(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step limit must be an integer.") from exc
    if limit == 0:
        raise argparse.ArgumentTypeError("step limit must be positive, or -1 for unlimited.")
    return limit


def parse_nonnegative_int_arg(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer.") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer.")
    return number


def parse_positive_int_arg(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MixUp/FMIX/SaliencyMix/Guided-SR PyTorch/XLA trainer")
    parser.add_argument("--config", default=None, help="YAML/JSON config path, e.g. configs/cifar10/preact_resnet18/fmix.yaml.")
    parser.add_argument("--dataset", choices=sorted({*DATASETS, *DATASET_ALIASES}), default=None)
    parser.add_argument("--recipe", choices=sorted(RECIPES), default=None)
    parser.add_argument("--method", choices=METHOD_CHOICES, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--download", action="store_true", default=None, help="Download CIFAR datasets if needed.")
    parser.add_argument("--no-augment", action="store_true", default=None, help="Disable train-time spatial augmentations.")
    parser.add_argument(
        "--aug-recipe",
        default=None,
        choices=["none", "basic", "hflip", "horizontal_flip", "imagenet", "tiny_official", "tiny_openmixup"],
        help="Explicit sample-level train augmentation recipe.",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", "--lr-schedule", dest="scheduler", choices=["cosine", "multistep", "step"], default=None)
    parser.add_argument("--milestones", "--lr-decay-epochs", dest="milestones", type=int, nargs="*", default=None)

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
    parser.add_argument(
        "--sal-aug-recipe",
        default=None,
        choices=["none", "basic", "hflip", "horizontal_flip", "imagenet", "tiny_official", "tiny_openmixup"],
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xla"], default="auto")
    parser.add_argument("--num-cores", type=parse_positive_int_arg, default=1, help="XLA processes to spawn when --device xla.")
    parser.add_argument("--num-workers", type=parse_nonnegative_int_arg, default=4)
    parser.add_argument("--seed", type=parse_seed_arg, default=None)
    parser.add_argument("--log-interval", type=parse_nonnegative_int_arg, default=50)
    parser.add_argument(
        "--max-train-steps",
        type=parse_step_limit_arg,
        default=None,
        help="Limit steps per epoch for smoke tests; use -1 for unlimited.",
    )
    parser.add_argument(
        "--max-val-steps",
        "--max-eval-steps",
        dest="max_val_steps",
        type=parse_step_limit_arg,
        default=None,
        help="Limit validation/evaluation steps for smoke tests; use -1 for unlimited.",
    )
    parser.add_argument(
        "--checkpoint",
        "--resume-checkpoint",
        dest="checkpoint",
        default=None,
        help="Load a model checkpoint before training/evaluation.",
    )
    parser.add_argument(
        "--final-test-checkpoint",
        choices=["last", "best"],
        default=None,
        help="Weights used for the final test split after training.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the configured validation/final-test splits, usually with --checkpoint.",
    )
    parser.add_argument("--save-every", type=parse_nonnegative_int_arg, default=0, help="Save periodic epoch checkpoints; 0 disables.")
    return parser.parse_args(argv)


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


def _first_section(config: dict[str, Any], names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        section = _section(config, name)
        if section:
            return section
    return {}


def _method_section(config: dict[str, Any], method_name: str, raw_method_name: Any | None = None) -> dict[str, Any]:
    names = method_section_aliases(method_name)
    if raw_method_name is not None and normalize_method_name(raw_method_name) == normalize_method_name(method_name):
        names.insert(1, str(raw_method_name).lower())
    return _first_section(config, _unique_strings(names))


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


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "on"}:
            return True
        if normalized in {"false", "no", "n", "0", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _normalize_step_limit(value: Any, name: str) -> int | None:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if limit == 0:
        raise ValueError(f"{name} must be positive, or -1 for unlimited; got {value}.")
    return None if limit < 0 else limit


def _config_limit(raw_config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = raw_config.get(key)
        if value is None:
            continue
        return _normalize_step_limit(value, key)
    return None


def resolve_step_limit(cli_value: int | None, raw_config: dict[str, Any], *keys: str) -> int | None:
    if cli_value is not None:
        return _normalize_step_limit(cli_value, keys[0] if keys else "step limit")
    return _config_limit(raw_config, *keys)


def relocate_relative_saliency_path(
    saliency_path: str | Path | None,
    saliency_dir: str | Path | None,
) -> str | None:
    """Move a relative cache filename under a CLI-provided saliency directory."""

    if saliency_path in ("", None):
        return None
    path = Path(str(saliency_path))
    if saliency_dir in ("", None) or path.is_absolute():
        return str(saliency_path)
    return (Path(str(saliency_dir)) / path.name).as_posix()


def default_guided_sr_saliency_path(dataset_name: str, saliency_dir: str | Path) -> str:
    dataset_stem = "tiny_imagenet" if normalize_dataset_name(dataset_name) == "tinyimagenet" else normalize_dataset_name(dataset_name)
    return (Path(str(saliency_dir)) / f"{dataset_stem}_train_guided_sr_saliency.npy").as_posix()


def _path_values_equal(left: str | Path | None, right: str | Path | None) -> bool:
    if left in ("", None) or right in ("", None):
        return left in ("", None) and right in ("", None)
    return Path(str(left)) == Path(str(right))


def resolve_saliency_storage_paths(
    raw_config: dict[str, Any],
    data_dir_override: str | Path | None = None,
    saliency_dir_override: str | Path | None = None,
    saliency_path_override: str | Path | None = None,
) -> tuple[str, str | None]:
    raw_data_dir = raw_config.get("data_dir", "./data")
    data_dir = data_dir_override or raw_data_dir
    raw_saliency_dir = raw_config.get("saliency_dir")
    default_saliency_dir = raw_saliency_dir or raw_data_dir
    saliency_dir_follows_data_dir = _path_values_equal(default_saliency_dir, raw_data_dir)
    saliency_dir = (
        saliency_dir_override
        or (data_dir if data_dir_override is not None and saliency_dir_follows_data_dir else default_saliency_dir)
    )

    if saliency_path_override is not None:
        saliency_path = relocate_relative_saliency_path(saliency_path_override, saliency_dir_override)
    else:
        relocation_dir = saliency_dir if not _path_values_equal(saliency_dir, default_saliency_dir) else None
        saliency_path = relocate_relative_saliency_path(raw_config.get("saliency_path"), relocation_dir)
    return str(saliency_dir), saliency_path


def resolved_config(args: argparse.Namespace, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_config = raw_config or {}
    dataset_name = normalize_dataset_name(args.dataset or raw_config.get("dataset", "cifar10"))
    recipe_name = args.recipe or raw_config.get("recipe", "openmixup")
    raw_method_name = args.method or raw_config.get("method", "fmix")
    method_name = normalize_method_name(raw_method_name)
    dataset = get_dataset_preset(dataset_name)
    recipe = get_recipe_preset(dataset_name, recipe_name)
    method_section = _method_section(raw_config, method_name, raw_method_name)
    fmix_section = _method_section(raw_config, "fmix")
    guidedmixup_section = _method_section(raw_config, "guided_sr")
    saliencymix_section = _method_section(raw_config, "saliencymix")
    resizemix_section = _method_section(raw_config, "resizemix")
    saliency_dir_override = getattr(args, "saliency_dir", None)
    saliency_path_override = getattr(args, "saliency_path", None)
    data_dir_override = getattr(args, "data_dir", None)
    saliency_dir, saliency_path = resolve_saliency_storage_paths(
        raw_config,
        data_dir_override=data_dir_override,
        saliency_dir_override=saliency_dir_override,
        saliency_path_override=saliency_path_override,
    )

    basic_aug_explicit = any(key in raw_config for key in ("use_basic_augmentation", "augment", "basic_aug"))
    image_aug_recipe_explicit = getattr(args, "aug_recipe", None) is not None or raw_config.get("aug_recipe") is not None
    config_augment = raw_config.get(
        "use_basic_augmentation",
        raw_config.get("augment", raw_config.get("basic_aug", True)),
    )
    use_basic_augmentation = _as_bool(config_augment, "basic_aug")
    if args.no_augment is True:
        use_basic_augmentation = False
    augmentation_recipe = getattr(args, "aug_recipe", None) or raw_config.get("aug_recipe") or None
    if args.no_augment is True:
        augmentation_recipe = "none"
    sal_basic_aug = _as_bool(raw_config.get("sal_basic_aug", False), "sal_basic_aug")
    sal_aug_recipe = (
        getattr(args, "sal_aug_recipe", None)
        or raw_config.get("sal_aug_recipe")
        or ("basic" if sal_basic_aug else "none")
    )
    if args.no_augment is True:
        sal_basic_aug = False
        sal_aug_recipe = "none"

    method_has_mixer = method_name not in {"baseline", "none", "eval"}
    method_prob_keys = _unique_strings(
        _method_prefixed_keys(method_name, "prob")
        + (["method_prob", "mix_prob"] if method_has_mixer else ["method_prob"])
    )
    method_alpha_keys = _unique_strings(
        _method_prefixed_keys(method_name, "alpha")
        + (["mix_alpha"] if method_has_mixer else [])
    )

    if not method_has_mixer:
        method_prob = 1.0
        method_alpha = recipe.alpha
    elif args.mix_prob is not None:
        method_prob = args.mix_prob
        method_alpha = args.alpha
    elif args.fmix_prob is not None and _method_is_active(method_name, "fmix"):
        method_prob = args.fmix_prob
        method_alpha = args.alpha
    else:
        method_prob = method_section.get(
            "prob",
            _first_config_value(raw_config, method_prob_keys, 1.0),
        )
        method_alpha = args.alpha
    if method_alpha is None:
        method_alpha = method_section.get(
            "alpha",
            _first_config_value(raw_config, method_alpha_keys, recipe.alpha),
        )

    if _method_is_active(method_name, "guided_sr"):
        guidedmixup_blur_kernel = (
            getattr(args, "guidedmixup_blur_kernel", None)
            if getattr(args, "guidedmixup_blur_kernel", None) is not None
            else _method_specific_value(
                raw_config,
                method_name,
                "guided_sr",
                guidedmixup_section,
                "blur_kernel",
                7,
            )
        )
        guidedmixup_condition = (
            getattr(args, "guidedmixup_condition", None)
            if getattr(args, "guidedmixup_condition", None) is not None
            else _method_specific_value(
                raw_config,
                method_name,
                "guided_sr",
                guidedmixup_section,
                "condition",
                "greedy",
            )
        )
    else:
        guidedmixup_blur_kernel = 7
        guidedmixup_condition = "greedy"

    if _method_is_active(method_name, "saliencymix"):
        saliency_source = (
            getattr(args, "saliency_source", None)
            or _method_specific_value(
                raw_config,
                method_name,
                "saliencymix",
                saliencymix_section,
                "saliency_source",
                raw_config.get("saliency_source", "spectral_residual"),
            )
        )
    elif _method_is_active(method_name, "guided_sr"):
        saliency_source = (
            getattr(args, "saliency_source", None)
            or _method_specific_value(
                raw_config,
                method_name,
                "guided_sr",
                guidedmixup_section,
                "saliency_source",
                raw_config.get("saliency_source", "spectral_residual"),
            )
        )
    else:
        saliency_source = "spectral_residual"

    if (
        method_name in {"saliencymix", "guided_sr"}
        and getattr(args, "saliency_source", None) is not None
        and str(saliency_source).lower() == "batch"
        and augmentation_recipe not in (None, "")
        and str(augmentation_recipe).lower() != "none"
    ):
        if str(sal_aug_recipe).lower() in {"", "none"}:
            sal_aug_recipe = augmentation_recipe
        augmentation_recipe = "none"

    if _method_is_active(method_name, "guided_sr") and str(saliency_source).lower() == "batch" and saliency_path is None:
        saliency_path = default_guided_sr_saliency_path(dataset_name, saliency_dir)

    if (
        method_name in {"saliencymix", "guided_sr"}
        and str(saliency_source).lower() != "batch"
        and not use_basic_augmentation
        and augmentation_recipe is None
        and not image_aug_recipe_explicit
        and args.no_augment is not True
        and str(sal_aug_recipe).lower() != "none"
    ):
        augmentation_recipe = sal_aug_recipe

    final_test_checkpoint = str(
        getattr(args, "final_test_checkpoint", None) or raw_config.get("final_test_checkpoint", "last")
    ).lower()
    if final_test_checkpoint not in {"last", "best"}:
        raise ValueError(f"final_test_checkpoint must be 'last' or 'best', got {final_test_checkpoint!r}.")

    model_name = canonical_model_name(str(raw_config.get("model", "preact_resnet18")))

    config = {
        "dataset": dataset_name,
        "recipe": recipe_name,
        "model": model_name,
        "model_impl_version": model_impl_version(model_name),
        "method": method_name,
        "data_dir": args.data_dir or raw_config.get("data_dir", "./data"),
        "output_dir": args.output_dir or raw_config.get("output_dir", "./runs/fmix"),
        "checkpoint": args.checkpoint or raw_config.get("checkpoint") or raw_config.get("resume_checkpoint") or None,
        "download": _as_bool(args.download if args.download is not None else raw_config.get("download", False), "download"),
        "use_basic_augmentation": use_basic_augmentation,
        "aug_recipe": augmentation_recipe,
        "num_classes": int(raw_config.get("num_classes", dataset.num_classes)),
        "image_size": dataset.image_size,
        "mean": dataset.mean,
        "std": dataset.std,
        "epochs": _choose(args, raw_config, "epochs", "training", "epochs", recipe.epochs),
        "batch_size": _choose(args, raw_config, "batch_size", "training", "batch_size", recipe.batch_size),
        "global_batch_size": raw_config.get("global_batch_size"),
        "lr": _choose(args, raw_config, "lr", "training", "lr", raw_config.get("learning_rate", recipe.lr)),
        "momentum": _choose(args, raw_config, "momentum", "training", "momentum", recipe.momentum),
        "weight_decay": _choose(args, raw_config, "weight_decay", "training", "weight_decay", recipe.weight_decay),
        "scheduler": _normalize_scheduler_name(
            _choose(args, raw_config, "scheduler", "training", "scheduler", raw_config.get("lr_schedule", recipe.scheduler))
        ),
        "milestones": validate_milestones(
            _choose(
                args,
                raw_config,
                "milestones",
                "training",
                "milestones",
                raw_config.get("lr_decay_epochs", list(recipe.milestones)),
            )
        ),
        "lr_decay_rate": float(raw_config.get("lr_decay_rate", 0.1)),
        "min_learning_rate": float(raw_config.get("min_learning_rate", 0.0)),
        "alpha": method_alpha,
        "decay_power": (
            args.decay_power
            if args.decay_power is not None and _method_is_active(method_name, "fmix")
            else _method_specific_value(
                raw_config,
                method_name,
                "fmix",
                fmix_section,
                "decay_power",
                recipe.decay_power,
                extra_keys=("decay_power", "fmix_decay"),
            )
        ),
        "max_soft": (
            args.max_soft
            if args.max_soft is not None and _method_is_active(method_name, "fmix")
            else _method_specific_value(
                raw_config,
                method_name,
                "fmix",
                fmix_section,
                "max_soft",
                recipe.max_soft,
                extra_keys=("max_soft",),
            )
        ),
        "transform_profile": recipe.transform_profile,
        "reformulate": _as_bool(
            args.reformulate
            if args.reformulate is not None and _method_is_active(method_name, "fmix")
            else _method_specific_value(
                raw_config,
                method_name,
                "fmix",
                fmix_section,
                "reformulate",
                False,
                extra_keys=("reformulate",),
            ),
            "reformulate",
        ),
        "method_prob": method_prob,
        "fmix_prob": method_prob,
        "guidedmixup_prob": method_prob,
        "saliencymix_prob": method_prob,
        "cutmix_prob": method_prob,
        "resizemix_prob": method_prob,
        "catchupmix_prob": method_prob,
        "mixup_no_repeat": _as_bool(
            _method_specific_value(raw_config, method_name, "mixup", method_section, "no_repeat", False),
            "mixup_no_repeat",
        ),
        "fmix_no_repeat": _as_bool(
            _method_specific_value(raw_config, method_name, "fmix", fmix_section, "no_repeat", False),
            "fmix_no_repeat",
        ),
        "cutmix_no_repeat": _as_bool(
            _method_specific_value(raw_config, method_name, "cutmix", method_section, "no_repeat", False),
            "cutmix_no_repeat",
        ),
        "catchupmix_cutmix_alpha": float(
            _method_specific_value(
                raw_config,
                method_name,
                "catchupmix",
                method_section,
                "cutmix_alpha",
                1.0,
            )
        ),
        "catchupmix_num_layers": int(
            _method_specific_value(
                raw_config,
                method_name,
                "catchupmix",
                method_section,
                "num_layers",
                5,
            )
        ),
        "catchupmix_no_repeat": _as_bool(
            _method_specific_value(raw_config, method_name, "catchupmix", method_section, "no_repeat", False),
            "catchupmix_no_repeat",
        ),
        "resizemix_scope_min": float(
            _method_specific_value(
                raw_config,
                method_name,
                "resizemix",
                resizemix_section,
                "scope_min",
                0.1,
            )
        ),
        "resizemix_scope_max": float(
            _method_specific_value(
                raw_config,
                method_name,
                "resizemix",
                resizemix_section,
                "scope_max",
                0.8,
            )
        ),
        "resizemix_use_alpha": _as_bool(
            _method_specific_value(
                raw_config,
                method_name,
                "resizemix",
                resizemix_section,
                "use_alpha",
                False,
            ),
            "resizemix_use_alpha",
        ),
        "resizemix_no_repeat": _as_bool(
            _method_specific_value(
                raw_config,
                method_name,
                "resizemix",
                resizemix_section,
                "no_repeat",
                False,
            ),
            "resizemix_no_repeat",
        ),
        "saliencymix_no_repeat": _as_bool(
            _method_specific_value(
                raw_config,
                method_name,
                "saliencymix",
                saliencymix_section,
                "no_repeat",
                False,
            ),
            "saliencymix_no_repeat",
        ),
        "guidedmixup_blur_kernel": int(guidedmixup_blur_kernel),
        "guidedmixup_condition": str(guidedmixup_condition).lower(),
        "saliency_source": str(saliency_source).lower(),
        "saliency_dir": saliency_dir,
        "saliency_path": saliency_path,
        "sal_basic_aug": sal_basic_aug,
        "sal_aug_recipe": sal_aug_recipe,
        "validate_saliency_cache_on_load": _as_bool(
            raw_config.get("validate_saliency_cache_on_load", False),
            "validate_saliency_cache_on_load",
        ),
        "cross_device_shuffle": _as_bool(raw_config.get("cross_device_shuffle", False), "cross_device_shuffle"),
        "validation_split": float(raw_config.get("validation_split", 0.0)),
        "eval_on_test_each_epoch": _as_bool(raw_config.get("eval_on_test_each_epoch", True), "eval_on_test_each_epoch"),
        "final_test": _as_bool(raw_config.get("final_test", False), "final_test"),
        "final_test_checkpoint": final_test_checkpoint,
        "run_name": raw_config.get("run_name", ""),
        "save_csv": _as_bool(raw_config.get("save_csv", False), "save_csv"),
        "run_metadata_required": _as_bool(raw_config.get("run_metadata_required", False), "run_metadata_required"),
        "output_name": raw_config.get("output_name", ""),
        "save_checkpoint": _as_bool(raw_config.get("save_checkpoint", True), "save_checkpoint"),
        "save_best_only": _as_bool(raw_config.get("save_best_only", False), "save_best_only"),
        "checkpoint_dir": args.checkpoint_dir if getattr(args, "checkpoint_dir", None) is not None else raw_config.get("checkpoint_dir"),
    }
    if (
        method_uses_batch_saliency(config)
        and not basic_aug_explicit
        and not image_aug_recipe_explicit
        and args.no_augment is not True
    ):
        config["use_basic_augmentation"] = False
    validate_resolved_config(config)
    return config


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _as_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


def validate_probability(name: str, value: Any) -> float:
    probability = _as_float(value, name)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return probability


def validate_positive_float(name: str, value: Any) -> float:
    number = _as_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return number


def validate_nonnegative_float(name: str, value: Any) -> float:
    number = _as_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return number


def validate_positive_int(name: str, value: Any) -> int:
    number = _as_int(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return number


def validate_odd_positive_int(name: str, value: Any) -> int:
    number = validate_positive_int(name, value)
    if number % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}.")
    return number


def validate_milestones(value: Any) -> list[int]:
    if value in ("", None):
        return []
    if isinstance(value, (str, bytes)):
        raise ValueError(f"milestones must be a sequence of nonnegative integers, got {value!r}.")
    try:
        milestones = [_as_int(item, "milestones") for item in value]
    except TypeError as exc:
        raise ValueError(f"milestones must be a sequence of nonnegative integers, got {value!r}.") from exc
    if any(milestone < 0 for milestone in milestones):
        raise ValueError(f"milestones must be nonnegative, got {milestones}.")
    return milestones


def uses_unpaired_image_augmentation(config: dict[str, Any]) -> bool:
    aug_recipe = config.get("aug_recipe")
    if aug_recipe not in (None, ""):
        return str(aug_recipe).lower() != "none"
    return bool(config.get("use_basic_augmentation", False))


def validate_resolved_config(config: dict[str, Any]) -> None:
    method = str(config.get("method", "")).lower()
    if _as_int(config.get("epochs"), "epochs") < 0:
        raise ValueError(f"epochs must be >= 0, got {config.get('epochs')}.")
    validate_positive_int("batch_size", config.get("batch_size"))
    validate_positive_int("num_classes", config.get("num_classes"))
    validate_positive_int("image_size", config.get("image_size"))
    validate_nonnegative_float("lr", config.get("lr"))
    validate_nonnegative_float("momentum", config.get("momentum"))
    validate_nonnegative_float("weight_decay", config.get("weight_decay"))
    validate_nonnegative_float("lr_decay_rate", config.get("lr_decay_rate"))
    validate_nonnegative_float("min_learning_rate", config.get("min_learning_rate"))
    validate_milestones(config.get("milestones"))
    validate_probability("method_prob", config.get("method_prob"))
    validation_split = _as_float(config.get("validation_split"), "validation_split")
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError(f"validation_split must be in [0, 1), got {config.get('validation_split')}.")
    global_batch_size = config.get("global_batch_size")
    if global_batch_size not in ("", None) and _as_int(global_batch_size, "global_batch_size") <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}.")
    if method in {"mixup", "fmix"}:
        validate_nonnegative_float("alpha", config.get("alpha"))
    elif method not in {"baseline", "none", "eval"}:
        validate_positive_float("alpha", config.get("alpha"))
    if method == "fmix":
        validate_positive_float("decay_power", config.get("decay_power"))
        validate_probability("max_soft", config.get("max_soft"))
    if method == "catchupmix":
        validate_positive_float("catchupmix_cutmix_alpha", config.get("catchupmix_cutmix_alpha"))
        validate_positive_int("catchupmix_num_layers", config.get("catchupmix_num_layers"))
    if method == "resizemix":
        scope_min = _as_float(config.get("resizemix_scope_min"), "resizemix_scope_min")
        scope_max = _as_float(config.get("resizemix_scope_max"), "resizemix_scope_max")
        if not (0.0 < scope_min <= scope_max <= 1.0):
            raise ValueError(
                "ResizeMix scope must satisfy 0 < resizemix_scope_min <= "
                f"resizemix_scope_max <= 1, got {scope_min}, {scope_max}."
            )
    if method in {"saliencymix", "guided_sr"}:
        validate_odd_positive_int("guidedmixup_blur_kernel", config.get("guidedmixup_blur_kernel"))
        if str(config.get("guidedmixup_condition", "")).lower() not in {"random", "greedy"}:
            raise ValueError(
                "guidedmixup_condition must be one of: random, greedy. "
                f"Got {config.get('guidedmixup_condition')}."
            )
    if method in {"saliencymix", "guided_sr"} and str(config.get("saliency_source", "")).lower() not in {
        "batch",
        "gradient",
        "grad",
        "spectral_residual",
        "guided_sr",
        "sr",
        "online",
    }:
        raise ValueError(f"Unsupported saliency_source: {config.get('saliency_source')}.")
    if method_uses_batch_saliency(config) and uses_unpaired_image_augmentation(config):
        raise ValueError(
            "Batch-saliency methods require basic_aug: false so cached saliency maps stay aligned with images. "
            "Use sal_aug_recipe for paired image/saliency spatial augmentation."
        )


def validate_global_batch_size(config: dict[str, Any], world_size: int, use_xla: bool) -> None:
    expected = config.get("global_batch_size")
    if expected in ("", None):
        return
    if not use_xla:
        return
    expected_batch = int(expected)
    actual_batch = int(config["batch_size"]) * int(world_size)
    if actual_batch != expected_batch:
        raise ValueError(
            "Configured global_batch_size does not match the XLA launch: "
            f"batch_size={config['batch_size']} * world_size={world_size} = {actual_batch}, "
            f"expected {expected_batch}. Use the xla4 scripts or adjust batch_size/global_batch_size together."
        )


def method_uses_batch_saliency(config: dict[str, Any]) -> bool:
    return (
        str(config["method"]) in {"saliencymix", "guided_sr"}
        and str(config["saliency_source"]) == "batch"
    )


def training_needs_batch_saliency_maps(config: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        method_uses_batch_saliency(config)
        and not bool(getattr(args, "eval_only", False))
        and int(config["epochs"]) > 0
    )


def validate_seed(seed: int | None, name: str = "--seed") -> int:
    value = int(seed or 0)
    if value < 0 or value > MAX_NUMPY_SEED:
        raise ValueError(f"{name} must be in [0, {MAX_NUMPY_SEED}], got {seed}.")
    return value


def derive_seed(seed: int | None, rank: int = 0, offset: int = 0) -> int:
    return (validate_seed(seed) + int(rank) * 1_000_003 + int(offset)) % (MAX_NUMPY_SEED + 1)


def set_seed(seed: int) -> None:
    seed = validate_seed(seed, "seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_data_loader_generator(seed: int | None, rank: int = 0, offset: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(derive_seed(seed, rank=rank, offset=offset))


def data_loader_seed_kwargs(seed: int | None, rank: int = 0, offset: int = 0) -> dict[str, Any]:
    return {
        "worker_init_fn": seed_worker,
        "generator": make_data_loader_generator(seed, rank=rank, offset=offset),
    }


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
        augmentation_recipe="none",
        normalize_train=True,
    )

    num_total = len(train_set)
    if len(eval_train_set) != num_total:
        raise ValueError(
            "validation_split requires train and eval-train dataset views to have the same length: "
            f"train={num_total}, eval_train={len(eval_train_set)}."
        )
    num_val = max(1, int(round(num_total * split)))
    num_train = num_total - num_val
    if num_train <= 0:
        raise ValueError(
            "validation_split leaves no training examples: "
            f"dataset_size={num_total}, validation_split={split}, val_examples={num_val}."
        )
    generator = torch.Generator().manual_seed(validate_seed(seed))
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
    "eval_top5_accuracy",
    "eval_top5_error",
    "val_loss",
    "val_top1",
    "val_top1_error",
    "val_top5",
    "val_top5_error",
    "best_top1_error",
    "best_epoch",
    "best_top1",
    "test_loss",
    "test_top1_accuracy",
    "test_top1_error",
    "test_top1",
    "test_top5_accuracy",
    "test_top5_error",
    "test_top5",
    "final_test_checkpoint",
    "final_test_checkpoint_source",
]


RUN_METADATA_COMPATIBILITY_KEYS = (
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


def metrics_csv_path(run_dir: Path, config: dict[str, Any]) -> Path:
    output_name = str(config.get("output_name") or "").strip()
    filename = f"{output_name}.csv" if output_name else "metrics.csv"
    return run_dir / filename


def _compatible_metadata_value(actual: object, expected: object) -> bool:
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


def _compatible_metadata_field(key: str, actual: object, expected: object) -> bool:
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
    return _compatible_metadata_value(actual, expected)


def load_run_metadata_config(path: Path) -> dict[str, Any] | None:
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


def run_metadata_matches_config(metadata_config: dict[str, Any] | None, config: dict[str, Any]) -> bool:
    if not isinstance(metadata_config, dict):
        return False
    for key in RUN_METADATA_COMPATIBILITY_KEYS:
        if key not in metadata_config:
            return False
        if not _compatible_metadata_field(key, metadata_config.get(key), config.get(key)):
            return False
    return True


def archive_run_artifact(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}.stale-{stamp}{path.suffix}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.stale-{stamp}-{suffix}{path.suffix}")
        suffix += 1
    path.replace(candidate)
    return candidate


def temporary_run_metadata_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def write_run_metadata_atomic(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_run_metadata_path(path)
    temp_path.unlink(missing_ok=True)
    try:
        temp_path.write_text(json.dumps(metadata, indent=2))
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def prepare_metrics_csv_for_run(
    csv_path: Path,
    config_path: Path,
    config: dict[str, Any],
    archive_compatible: bool = False,
) -> Path | None:
    if not bool(config.get("run_metadata_required", False)):
        return None
    if not csv_path.exists():
        return None
    metadata_config = load_run_metadata_config(config_path)
    if run_metadata_matches_config(metadata_config, config) and not bool(archive_compatible):
        return None
    archived_csv = archive_run_artifact(csv_path)
    archive_run_artifact(config_path)
    return archived_csv


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

    def normalize_accuracy_fraction(accuracy_field: str) -> None:
        accuracy = _maybe_float(normalized.get(accuracy_field))
        if accuracy is not None and accuracy > 1.0:
            normalized[accuracy_field] = f"{pct_to_fraction(accuracy):.6f}"

    fill_from_percent("train_top1", "train_accuracy", "train_top1_error")
    fill_from_percent("val_top1", "eval_top1_accuracy", "eval_top1_error")
    fill_from_percent("val_top5", "eval_top5_accuracy", "eval_top5_error")
    fill_from_percent("test_top1", "test_top1_accuracy", "test_top1_error")
    fill_from_percent("test_top5", "test_top5_accuracy", "test_top5_error")

    val_top1 = _maybe_float(row.get("val_top1"))
    if val_top1 is not None and not normalized.get("val_top1_error"):
        normalized["val_top1_error"] = f"{pct_to_error(val_top1):.6f}"
    val_top5 = _maybe_float(row.get("val_top5"))
    if val_top5 is not None and not normalized.get("val_top5_error"):
        normalized["val_top5_error"] = f"{pct_to_error(val_top5):.6f}"

    best_top1 = _maybe_float(row.get("best_top1"))
    if best_top1 is not None and not normalized.get("best_top1_error"):
        normalized["best_top1_error"] = f"{pct_to_error(best_top1):.6f}"

    normalize_accuracy_fraction("train_accuracy")
    normalize_accuracy_fraction("eval_top1_accuracy")
    normalize_accuracy_fraction("eval_top5_accuracy")
    normalize_accuracy_fraction("test_top1_accuracy")
    normalize_accuracy_fraction("test_top5_accuracy")

    fill_error_from_accuracy("train_accuracy", "train_top1_error")
    fill_error_from_accuracy("eval_top1_accuracy", "eval_top1_error")
    fill_error_from_accuracy("eval_top5_accuracy", "eval_top5_error")
    fill_error_from_accuracy("test_top1_accuracy", "test_top1_error")
    fill_error_from_accuracy("test_top5_accuracy", "test_top5_error")

    return normalized


def temporary_metrics_csv_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def read_normalized_metrics_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [normalize_metrics_row(row) for row in reader]


def write_metrics_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_metrics_csv_path(path)
    temp_path.unlink(missing_ok=True)
    try:
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def migrate_metrics_csv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == CSV_FIELDS:
            return
        rows = [normalize_metrics_row(row) for row in reader]
    write_metrics_csv_atomic(path, rows)


def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_normalized_metrics_rows(path)
    rows.append(normalize_metrics_row(row))
    write_metrics_csv_atomic(path, rows)


def append_eval_metrics_csv(
    path: Path,
    epoch: int,
    val_loss: float,
    val_acc: float,
    val_top5: float,
    best_acc: float,
    best_epoch: int,
) -> None:
    val_error = pct_to_error(val_acc)
    val_top5_error = pct_to_error(val_top5)
    append_metrics_csv(
        path,
        {
            "epoch": epoch,
            "phase": "eval",
            "eval_loss": f"{val_loss:.6f}",
            "eval_top1_accuracy": f"{pct_to_fraction(val_acc):.6f}",
            "eval_top1_error": f"{val_error:.6f}",
            "eval_top5_accuracy": f"{pct_to_fraction(val_top5):.6f}",
            "eval_top5_error": f"{val_top5_error:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_top1": f"{val_acc:.6f}",
            "val_top1_error": f"{val_error:.6f}",
            "val_top5": f"{val_top5:.6f}",
            "val_top5_error": f"{val_top5_error:.6f}",
            "best_top1": f"{best_acc:.6f}",
            "best_top1_error": f"{pct_to_error(best_acc):.6f}",
            "best_epoch": best_epoch,
        },
    )


def append_final_test_metrics_csv(
    path: Path,
    epoch: int,
    test_loss: float,
    test_acc: float,
    test_top5: float,
    best_acc: float,
    best_epoch: int,
    final_test_checkpoint: str = "",
    final_test_checkpoint_source: str = "",
) -> None:
    test_error = pct_to_error(test_acc)
    test_top5_error = pct_to_error(test_top5)
    append_metrics_csv(
        path,
        {
            "epoch": epoch,
            "phase": "final_test",
            "best_top1": f"{best_acc:.6f}",
            "best_top1_error": f"{pct_to_error(best_acc):.6f}",
            "best_epoch": best_epoch,
            "test_loss": f"{test_loss:.6f}",
            "test_top1_accuracy": f"{pct_to_fraction(test_acc):.6f}",
            "test_top1_error": f"{test_error:.6f}",
            "test_top1": f"{test_acc:.6f}",
            "test_top5_accuracy": f"{pct_to_fraction(test_top5):.6f}",
            "test_top5_error": f"{test_top5_error:.6f}",
            "test_top5": f"{test_top5:.6f}",
            "final_test_checkpoint": final_test_checkpoint,
            "final_test_checkpoint_source": final_test_checkpoint_source,
        },
    )


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> int:
    return int(topk_correct_tensor(logits, targets, k).item())


def topk_correct_tensor(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> torch.Tensor:
    k = min(int(k), int(logits.size(1)))
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


def require_processed_samples(total: float, phase: str) -> float:
    if total <= 0:
        raise RuntimeError(
            f"{phase} processed 0 samples; check dataset size, batch_size/drop_last, "
            "validation_split, and max step limits."
        )
    return total


class SequentialDistributedEvalSampler(Sampler[int]):
    """Shard eval datasets without padding duplicate samples across ranks."""

    def __init__(self, dataset, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        dataset_len = len(self.dataset)
        if dataset_len <= self.rank:
            return 0
        return ((dataset_len - 1 - self.rank) // self.num_replicas) + 1


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


def checkpoint_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def archive_checkpoint_artifacts_for_fresh_run(checkpoint_root: Path) -> list[Path]:
    if not checkpoint_root.exists():
        return []

    candidates = [
        checkpoint_root / "best.pt",
        checkpoint_root / "last.pt",
        *sorted(checkpoint_root.glob("epoch_*.pt")),
    ]
    archived: list[Path] = []
    seen: set[Path] = set()
    for checkpoint_path in candidates:
        for path in (checkpoint_path, checkpoint_metadata_path(checkpoint_path)):
            normalized = path.resolve(strict=False)
            if normalized in seen:
                continue
            seen.add(normalized)
            archived_path = archive_run_artifact(path)
            if archived_path is not None:
                archived.append(archived_path)
    return archived


def checkpoint_backup_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.bak")


def temporary_checkpoint_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def temporary_checkpoint_metadata_path(path: Path) -> Path:
    metadata_path = checkpoint_metadata_path(path)
    return metadata_path.with_name(f".{metadata_path.name}.tmp")


def _recover_checkpoint_backup_if_needed(
    path: Path,
    metadata_path: Path,
    backup_path: Path,
    backup_metadata_path: Path,
) -> None:
    checkpoint_backup_exists = backup_path.exists()
    metadata_backup_exists = backup_metadata_path.exists()
    if not checkpoint_backup_exists and not metadata_backup_exists:
        return
    final_pair_complete = path.exists() and metadata_path.exists()
    backup_pair_complete = checkpoint_backup_exists and metadata_backup_exists
    if backup_pair_complete and not final_pair_complete:
        path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        backup_path.replace(path)
        backup_metadata_path.replace(metadata_path)
        return
    if checkpoint_backup_exists and not path.exists():
        backup_path.replace(path)
    if metadata_backup_exists and not metadata_path.exists():
        backup_metadata_path.replace(metadata_path)
    if final_pair_complete:
        backup_path.unlink(missing_ok=True)
        backup_metadata_path.unlink(missing_ok=True)


def save_checkpoint_payload_atomic(
    path: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
    use_xla: bool,
    xm: Any | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_metadata_path(path)
    temp_checkpoint_path = temporary_checkpoint_path(path)
    temp_metadata_path = temporary_checkpoint_metadata_path(path)
    backup_path = checkpoint_backup_path(path)
    backup_metadata_path = checkpoint_backup_path(metadata_path)
    _recover_checkpoint_backup_if_needed(path, metadata_path, backup_path, backup_metadata_path)
    for temp_path in (temp_checkpoint_path, temp_metadata_path, backup_path, backup_metadata_path):
        temp_path.unlink(missing_ok=True)

    checkpoint_backed_up = False
    metadata_backed_up = False
    checkpoint_installed = False
    metadata_installed = False
    try:
        if use_xla:
            xm.save(payload, str(temp_checkpoint_path))
        else:
            torch.save(payload, temp_checkpoint_path)
        temp_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        if path.exists():
            path.replace(backup_path)
            checkpoint_backed_up = True
        if metadata_path.exists():
            metadata_path.replace(backup_metadata_path)
            metadata_backed_up = True
        temp_checkpoint_path.replace(path)
        checkpoint_installed = True
        temp_metadata_path.replace(metadata_path)
        metadata_installed = True
        backup_path.unlink(missing_ok=True)
        backup_metadata_path.unlink(missing_ok=True)
        checkpoint_backed_up = False
        metadata_backed_up = False
    except Exception:
        if checkpoint_installed:
            path.unlink(missing_ok=True)
        if metadata_installed:
            metadata_path.unlink(missing_ok=True)
        if checkpoint_backed_up and backup_path.exists():
            backup_path.replace(path)
            checkpoint_backed_up = False
        if metadata_backed_up and backup_metadata_path.exists():
            backup_metadata_path.replace(metadata_path)
            metadata_backed_up = False
        for temp_path in (temp_checkpoint_path, temp_metadata_path):
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        if not checkpoint_backed_up:
            backup_path.unlink(missing_ok=True)
        if not metadata_backed_up:
            backup_metadata_path.unlink(missing_ok=True)


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
    best_epoch: int | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "best_acc": best_acc,
        "best_epoch": best_epoch if best_epoch is not None else epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    metadata = {
        "epoch": epoch,
        "best_acc": best_acc,
        "best_epoch": best_epoch if best_epoch is not None else epoch,
        "config": config,
    }
    save_checkpoint_payload_atomic(path, payload, metadata, use_xla=use_xla, xm=xm)


def _torch_load_checkpoint(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    checkpoint = _torch_load_checkpoint(path)
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


def validate_checkpoint_metadata_matches_config(
    path: str | Path,
    checkpoint_metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    require_config: bool = False,
) -> None:
    checkpoint_config = checkpoint_metadata.get("config") if isinstance(checkpoint_metadata, dict) else None
    if checkpoint_config is None:
        if require_config:
            raise RuntimeError(
                f"Checkpoint {path} has no config metadata. "
                "Refusing to use it for a metadata-required table run."
            )
        return
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError(f"Checkpoint {path} has invalid config metadata.")
    for key in RUN_METADATA_COMPATIBILITY_KEYS:
        if key not in checkpoint_config:
            raise RuntimeError(
                f"Checkpoint {path} is incompatible with the current run config: "
                f"missing metadata key {key!r}."
            )
        if not _compatible_metadata_field(key, checkpoint_config.get(key), config.get(key)):
            raise RuntimeError(
                f"Checkpoint {path} is incompatible with the current run config: "
                f"{key}={checkpoint_config.get(key)!r}, expected {config.get(key)!r}."
            )


def load_checkpoint_metadata_for_validation(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    sidecar_path = checkpoint_metadata_path(checkpoint_path)
    if sidecar_path.exists():
        try:
            metadata = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Checkpoint metadata sidecar is not readable JSON: {sidecar_path}") from exc
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Checkpoint metadata sidecar must contain a JSON object: {sidecar_path}")
        return metadata
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    return checkpoint if isinstance(checkpoint, dict) else {}


def checkpoint_metadata_for_validation(
    path: str | Path,
    loaded_checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sidecar_path = checkpoint_metadata_path(Path(path))
    if sidecar_path.exists():
        return load_checkpoint_metadata_for_validation(path)
    return loaded_checkpoint_metadata if isinstance(loaded_checkpoint_metadata, dict) else {}


def clone_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_model_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict(state_dict)
    model.to(device)


def restore_best_weights_for_final_test(
    model: torch.nn.Module,
    best_state_dict: dict[str, torch.Tensor] | None,
    best_checkpoint_path: Path,
    device: torch.device,
) -> str | None:
    if best_state_dict is not None:
        restore_model_state_dict(model, best_state_dict, device)
        return "memory"
    if best_checkpoint_path.exists():
        load_model_checkpoint(best_checkpoint_path, model)
        model.to(device)
        return best_checkpoint_path.as_posix()
    return None


def restore_required_best_weights_for_final_test(
    model: torch.nn.Module,
    best_state_dict: dict[str, torch.Tensor] | None,
    best_checkpoint_path: Path,
    device: torch.device,
) -> str:
    source = restore_best_weights_for_final_test(model, best_state_dict, best_checkpoint_path, device)
    if source is None:
        raise RuntimeError(
            "final_test_checkpoint=best requested, but no best weights were available. "
            f"Expected in-memory best weights or {best_checkpoint_path}."
        )
    return source


def _paths_match(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _is_named_best_checkpoint(path: Path | None) -> bool:
    return path is not None and path.name == "best.pt"


def _is_named_last_checkpoint(path: Path | None) -> bool:
    return path is not None and path.name == "last.pt"


def eval_only_best_checkpoint_to_load(
    config_checkpoint: str | Path | None,
    checkpoint_root: Path,
    explicit_checkpoint: bool = False,
) -> Path | None:
    default_best_path = checkpoint_root / "best.pt"
    loaded_checkpoint_path = Path(str(config_checkpoint)) if config_checkpoint else None
    if explicit_checkpoint and loaded_checkpoint_path is not None:
        if _is_named_last_checkpoint(loaded_checkpoint_path):
            if default_best_path.exists() and not _paths_match(loaded_checkpoint_path, default_best_path):
                return default_best_path
            raise RuntimeError(
                "Eval-only final_test_checkpoint=best requires a best checkpoint. "
                f"Got explicit last checkpoint {loaded_checkpoint_path}; pass a best checkpoint or use {default_best_path}."
            )
        return None
    if default_best_path.exists():
        if loaded_checkpoint_path is None or not _paths_match(loaded_checkpoint_path, default_best_path):
            return default_best_path
        return None
    if _is_named_best_checkpoint(loaded_checkpoint_path):
        return None
    raise RuntimeError(
        "Eval-only final_test_checkpoint=best requires a best checkpoint. "
        f"Expected {default_best_path} or pass --checkpoint /path/to/best.pt."
    )


def move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_training_state(
    checkpoint: dict[str, Any],
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> tuple[int, float, int, bool]:
    has_optimizer_state = checkpoint.get("optimizer") is not None
    has_scheduler_state = checkpoint.get("scheduler") is not None
    if not has_optimizer_state and not has_scheduler_state:
        return 1, 0.0, 0, False
    if scheduler is not None and has_optimizer_state != has_scheduler_state:
        missing = "scheduler" if has_optimizer_state else "optimizer"
        raise RuntimeError(
            f"Checkpoint has incomplete training state: missing {missing} state. "
            "Use a full AllTheMiXLA training checkpoint to resume, or pass a weight-only checkpoint for eval/fine-tune."
        )

    if has_optimizer_state:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None and has_scheduler_state:
        scheduler.load_state_dict(checkpoint["scheduler"])

    loaded_epoch = int(checkpoint.get("epoch") or 0)
    start_epoch = loaded_epoch + 1 if loaded_epoch > 0 else 1
    best_acc = float(checkpoint.get("best_acc") or 0.0)
    best_epoch = int(checkpoint.get("best_epoch") or loaded_epoch or 0)
    return start_epoch, best_acc, best_epoch, True


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
    saliency_mean = tuple(config["mean"]) if config.get("mean") else None
    saliency_std = tuple(config["std"]) if config.get("std") else None
    if method == "fmix":
        return FMix(
            decay_power=float(config["decay_power"]),
            alpha=float(config["alpha"]),
            size=(int(config["image_size"]), int(config["image_size"])),
            max_soft=float(config["max_soft"]),
            reformulate=bool(config["reformulate"]),
            no_repeat=bool(config.get("fmix_no_repeat", False)),
        )
    if method == "mixup":
        return MixUp(alpha=float(config["alpha"]), no_repeat=bool(config.get("mixup_no_repeat", False)))
    if method == "cutmix":
        return CutMix(alpha=float(config["alpha"]), no_repeat=bool(config.get("cutmix_no_repeat", False)))
    if method == "resizemix":
        return ResizeMix(
            scope_min=float(config.get("resizemix_scope_min", 0.1)),
            scope_max=float(config.get("resizemix_scope_max", 0.8)),
            alpha=float(config.get("alpha", 1.0)),
            use_alpha=bool(config.get("resizemix_use_alpha", False)),
            no_repeat=bool(config.get("resizemix_no_repeat", False)),
        )
    if method == "catchupmix":
        return CatchUpMix(
            alpha=float(config["alpha"]),
            cutmix_alpha=float(config.get("catchupmix_cutmix_alpha", 1.0)),
            num_feature_layers=int(config.get("catchupmix_num_layers", 5)),
            no_repeat=bool(config.get("catchupmix_no_repeat", False)),
        )
    if method == "saliencymix":
        return SaliencyMix(
            alpha=float(config["alpha"]),
            saliency_source=str(config.get("saliency_source", "spectral_residual")),
            blur_kernel=int(config.get("guidedmixup_blur_kernel", 7)),
            no_repeat=bool(config.get("saliencymix_no_repeat", False)),
            saliency_mean=saliency_mean,
            saliency_std=saliency_std,
        )
    if method == "guided_sr":
        return GuidedSR(
            alpha=float(config["alpha"]),
            blur_kernel=int(config.get("guidedmixup_blur_kernel", 7)),
            condition=str(config.get("guidedmixup_condition", "greedy")),
            saliency_mean=saliency_mean,
            saliency_std=saliency_std,
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
    if method in {"mixup", "cutmix", "resizemix", "catchupmix"}:
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
    no_repeat: bool = False,
    world_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local_batch = int(images.size(0))
    if local_batch <= 0:
        raise ValueError("cross_device_shuffle requires a non-empty local batch.")
    global_images = xm.all_gather(images, dim=0)
    global_targets = xm.all_gather(targets, dim=0)
    global_aux = xm.all_gather(aux_tensor, dim=0) if aux_tensor is not None else None

    local_scores = torch.rand(local_batch, device=images.device)
    global_scores = xm.all_gather(local_scores, dim=0)
    gathered_sizes = {
        "images": int(global_images.size(0)),
        "targets": int(global_targets.size(0)),
        "scores": int(global_scores.size(0)),
    }
    if global_aux is not None:
        gathered_sizes["aux"] = int(global_aux.size(0))
    if len(set(gathered_sizes.values())) != 1:
        detail = ", ".join(f"{name}={size}" for name, size in sorted(gathered_sizes.items()))
        raise ValueError(f"cross_device_shuffle gathered tensors must share batch dimension; got {detail}.")

    global_batch = int(global_images.size(0))
    if global_batch % local_batch != 0:
        raise ValueError(
            "cross_device_shuffle gathered global batch must be divisible by local batch: "
            f"global_batch={global_batch}, local_batch={local_batch}."
        )
    inferred_world_size = global_batch // local_batch
    if world_size is not None and int(world_size) != inferred_world_size:
        raise ValueError(
            "cross_device_shuffle world size mismatch: "
            f"world_size={world_size}, gathered global_batch/local_batch={inferred_world_size}."
        )
    if int(rank) < 0 or int(rank) >= inferred_world_size:
        raise ValueError(
            "cross_device_shuffle rank is outside gathered world size: "
            f"rank={rank}, inferred_world_size={inferred_world_size}."
        )

    global_index = torch.argsort(global_scores)
    if no_repeat and int(global_index.numel()) > 1:
        shifted_index = torch.roll(global_index, shifts=1, dims=0)
        no_repeat_index = torch.empty_like(global_index)
        no_repeat_index.scatter_(0, global_index, shifted_index)
        global_index = no_repeat_index

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


def resolve_guided_sr_saliency_maps(
    images: torch.Tensor,
    saliency_maps: torch.Tensor | None,
    config: dict[str, Any],
) -> torch.Tensor | None:
    if saliency_maps is not None:
        return saliency_maps
    source = str(config.get("saliency_source", "spectral_residual")).lower()
    if source in {"gradient", "grad"}:
        from allthemix.methods.saliencymix import compute_gradient_saliency_maps

        saliency_mean = tuple(config["mean"]) if config.get("mean") else None
        saliency_std = tuple(config["std"]) if config.get("std") else None
        saliency_images = denormalize_images_for_saliency(
            images,
            saliency_mean,
            saliency_std,
        )
        return compute_gradient_saliency_maps(saliency_images)
    return None


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
                    no_repeat=bool(config.get("saliencymix_no_repeat", False)),
                    world_size=world_size,
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
            partner_images, partner_targets, partner_index = cross_device_shuffle_batch(
                images,
                targets,
                rank,
                xm,
                no_repeat=bool(config.get("saliencymix_no_repeat", False)),
                world_size=world_size,
            )
            return mixer(
                images,
                targets,
                partner_images=partner_images,
                partner_targets=partner_targets,
                index=partner_index,
            )
        return mixer(images, targets, saliency_maps=saliency_maps)

    if method == "guided_sr":
        saliency_maps = resolve_guided_sr_saliency_maps(images, saliency_maps, config)
        if (
            use_xla
            and bool(config["cross_device_shuffle"])
            and world_size > 1
            and str(config.get("guidedmixup_condition", "greedy")).lower() == "random"
        ):
            if saliency_maps is not None:
                partner_images, partner_targets, partner_index, partner_saliency_maps = cross_device_shuffle_batch(
                    images,
                    targets,
                    rank,
                    xm,
                    aux_tensor=saliency_maps,
                    world_size=world_size,
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
            partner_images, partner_targets, partner_index = cross_device_shuffle_batch(
                images,
                targets,
                rank,
                xm,
                world_size=world_size,
            )
            return mixer(
                images,
                targets,
                saliency_maps=saliency_maps,
                partner_images=partner_images,
                partner_targets=partner_targets,
                index=partner_index,
            )
        return mixer(images, targets, saliency_maps=saliency_maps)

    if method == "catchupmix":
        return mixer(images, targets)

    if use_xla and bool(config["cross_device_shuffle"]) and world_size > 1:
        partner_images, partner_targets, partner_index = cross_device_shuffle_batch(
            images,
            targets,
            rank,
            xm,
            no_repeat=(
                (method == "cutmix" and bool(config.get("cutmix_no_repeat", False)))
                or (method == "resizemix" and bool(config.get("resizemix_no_repeat", False)))
                or (method == "mixup" and bool(config.get("mixup_no_repeat", False)))
                or (method == "fmix" and bool(config.get("fmix_no_repeat", False)))
            ),
            world_size=world_size,
        )
        return mixer(images, targets, partner_images, partner_targets, partner_index)
    return mixer(images, targets)


def should_apply_method(
    config: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    step: int,
    use_xla: bool,
    world_size: int,
) -> bool:
    prob = float(config["method_prob"])
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    if use_xla and world_size > 1:
        seed = int(args.seed or 0)
        rng = random.Random((seed + 1) * 1_000_003 + int(epoch) * 10_007 + int(step))
        return rng.random() < prob
    return random.random() < prob


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
        if mixer is not None and should_apply_method(config, args, epoch, step, use_xla, world_size):
            mixed = call_batch_mixer(mixer, images, targets, aux_info, config, use_xla, xm, rank, world_size)
            logits = model(mixed.images, feature_hook=getattr(mixed, "feature_hook", None))
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
        total_value = require_processed_samples(_tensor_float(total), "train")
        return _tensor_float(loss_sum) / total_value, 100.0 * _tensor_float(correct) / total_value

    loss_sum, correct, total = reduce_metrics((loss_sum, correct, total), f"train_{epoch}", use_xla, xm)
    total_value = require_processed_samples(float(total), "train")
    return loss_sum / total_value, 100.0 * correct / total_value


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
) -> tuple[float, float, float]:
    model.eval()
    if use_xla:
        loss_sum = torch.zeros((), device=device)
        correct_top1 = torch.zeros((), device=device)
        correct_top5 = torch.zeros((), device=device)
        total = torch.zeros((), device=device)
    else:
        loss_sum = 0.0
        correct_top1 = 0
        correct_top5 = 0
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
            correct_top1 = correct_top1 + topk_correct_tensor(logits, targets, k=1)
            correct_top5 = correct_top5 + topk_correct_tensor(logits, targets, k=5)
            total = total + batch_total
        else:
            loss_sum += float(loss.item()) * batch_size
            correct_top1 += topk_correct(logits, targets, k=1)
            correct_top5 += topk_correct(logits, targets, k=5)
            total += batch_size
        if use_xla:
            xm.mark_step()

    if use_xla:
        xm.mark_step()
        loss_sum, correct_top1, total = reduce_metric_tensors(loss_sum, correct_top1, total, f"val_{epoch}", use_xla, xm)
        correct_top5 = xm.mesh_reduce(f"val_{epoch}_correct_top5", correct_top5, _sum_tensors)
        total_value = require_processed_samples(_tensor_float(total), "validation/evaluation")
        return (
            _tensor_float(loss_sum) / total_value,
            100.0 * _tensor_float(correct_top1) / total_value,
            100.0 * _tensor_float(correct_top5) / total_value,
        )

    loss_sum, correct_top1, total = reduce_metrics((loss_sum, correct_top1, total), f"val_{epoch}", use_xla, xm)
    _, correct_top5, _ = reduce_metrics((0.0, correct_top5, total), f"val_{epoch}_top5", use_xla, xm)
    total_value = require_processed_samples(float(total), "validation/evaluation")
    return loss_sum / total_value, 100.0 * correct_top1 / total_value, 100.0 * correct_top5 / total_value


def evaluate_final_test(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    use_xla: bool,
    xm: Any | None,
    xr: Any | None,
    csv_path: Path,
    best_acc: float,
    best_epoch: int,
    checkpoint_source: str | None = None,
) -> tuple[float, float, float]:
    test_loss, test_acc, test_top5 = evaluate(model, test_loader, device, epoch, config, args, use_xla, xm)
    if bool(config["save_csv"]) and is_master(use_xla, xm, xr):
        append_final_test_metrics_csv(
            csv_path,
            epoch,
            test_loss,
            test_acc,
            test_top5,
            best_acc,
            best_epoch,
            final_test_checkpoint=str(config.get("final_test_checkpoint", "")),
            final_test_checkpoint_source=str(checkpoint_source or ""),
        )
    print_master(
        f"final_test_loss={test_loss:.4f} final_test_top1={test_acc:.2f} final_test_top5={test_top5:.2f}",
        use_xla,
        xm,
        xr,
    )
    return test_loss, test_acc, test_top5


def run_worker(index: int, args: argparse.Namespace) -> None:
    requested_xla = args.device == "xla"
    xla_modules = _optional_xla_import() if requested_xla else None
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
    args.seed = validate_seed(args.seed)
    set_seed(derive_seed(args.seed, rank=rank))
    args.max_train_steps = resolve_step_limit(args.max_train_steps, raw_config, "max_train_steps")
    args.max_val_steps = resolve_step_limit(args.max_val_steps, raw_config, "max_val_steps", "max_eval_steps")
    config = resolved_config(args, raw_config)
    validate_global_batch_size(config, world_size=world_size, use_xla=use_xla)
    preset = get_dataset_preset(config["dataset"])
    recipe = get_recipe_preset(config["dataset"], config["recipe"])
    needs_train_saliency_maps = training_needs_batch_saliency_maps(config, args)

    train_set, val_set = build_datasets(
        preset,
        recipe.transform_profile,
        data_dir=config["data_dir"],
        download=bool(config["download"]),
        use_basic_augmentation=bool(config["use_basic_augmentation"]),
        augmentation_recipe=config["aug_recipe"],
        normalize_train=not needs_train_saliency_maps,
    )
    if train_set is not None and needs_train_saliency_maps:
        train_set = attach_train_saliency_maps(
            train_set,
            dataset_name=str(config["dataset"]),
            saliency_dir=str(config["saliency_dir"]),
            saliency_path=config["saliency_path"],
            use_sal_basic_augmentation=bool(config["sal_basic_aug"]),
            saliency_augmentation_recipe=str(config["sal_aug_recipe"]),
            image_size=int(config["image_size"]),
            normalization_mean=tuple(config["mean"]),
            normalization_std=tuple(config["std"]),
            validate_finite=bool(config["validate_saliency_cache_on_load"]),
            validate_sample_finite=True,
        )
    train_set, val_set, test_set = apply_validation_split(train_set, val_set, preset, recipe, config, args.seed)
    if bool(config["eval_on_test_each_epoch"]) and test_set is not None:
        val_set = test_set

    distributed = world_size > 1
    should_build_train_loader = train_set is not None and not bool(args.eval_only) and int(config["epochs"]) > 0
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        if should_build_train_loader and distributed
        else None
    )
    val_sampler = (
        SequentialDistributedEvalSampler(val_set, num_replicas=world_size, rank=rank) if distributed else None
    )
    test_sampler = (
        SequentialDistributedEvalSampler(test_set, num_replicas=world_size, rank=rank)
        if test_set is not None and distributed
        else None
    )
    train_loader = None
    if should_build_train_loader:
        train_loader = DataLoader(
            train_set,
            batch_size=int(config["batch_size"]),
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            **data_loader_seed_kwargs(args.seed, rank=rank, offset=0),
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
        **data_loader_seed_kwargs(args.seed, rank=rank, offset=10_000),
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
            **data_loader_seed_kwargs(args.seed, rank=rank, offset=20_000),
        )

    if use_xla:
        if train_loader is not None:
            train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)
        if test_loader is not None:
            test_loader = pl.MpDeviceLoader(test_loader, device)

    model = build_model(str(config["model"]), num_classes=int(config["num_classes"]))
    checkpoint_meta: dict[str, Any] = {}
    if config["checkpoint"]:
        checkpoint_meta = load_model_checkpoint(str(config["checkpoint"]), model)
        validate_checkpoint_metadata_matches_config(
            config["checkpoint"],
            checkpoint_metadata_for_validation(config["checkpoint"], checkpoint_meta),
            config,
        )
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
    csv_path = metrics_csv_path(run_dir, config)
    if is_master(use_xla, xm, xr):
        run_dir.mkdir(parents=True, exist_ok=True)
        fresh_run_replaces_existing_artifacts = (
            not bool(args.eval_only)
            and train_loader is not None
            and int(config["epochs"]) > 0
            and not (
                bool(config["checkpoint"])
                and checkpoint_meta.get("optimizer") is not None
                and checkpoint_meta.get("scheduler") is not None
            )
        )
        archived_csv = prepare_metrics_csv_for_run(
            csv_path,
            run_dir / "config.json",
            config,
            archive_compatible=fresh_run_replaces_existing_artifacts,
        )
        if archived_csv is not None:
            print(
                f"Archived existing metrics before this run: {archived_csv}",
                flush=True,
            )
        if fresh_run_replaces_existing_artifacts:
            archived_checkpoints = archive_checkpoint_artifacts_for_fresh_run(checkpoint_root)
            if archived_checkpoints:
                print(
                    f"Archived {len(archived_checkpoints)} existing checkpoint artifacts before this run under "
                    f"{checkpoint_root}",
                    flush=True,
                )
        write_run_metadata_atomic(
            run_dir / "config.json",
            {
                "resolved": config,
                "preset": preset_dict(config["dataset"], config["recipe"]),
                "args": vars(args),
            },
        )
    print_master(
        f"Starting {str(config['method']).upper()} {config['dataset']}/{config['recipe']} on {device}; "
        f"world_size={world_size}; config={json.dumps(config, sort_keys=True)}",
        use_xla,
        xm,
        xr,
    )

    if args.eval_only or train_loader is None or int(config["epochs"]) <= 0:
        final_test_checkpoint_source = "current"
        if args.eval_only and str(config.get("final_test_checkpoint", "last")).lower() == "best":
            checkpoint_to_load = eval_only_best_checkpoint_to_load(
                config["checkpoint"],
                checkpoint_root,
                explicit_checkpoint=bool(args.checkpoint),
            )
            if checkpoint_to_load is not None:
                checkpoint_meta = load_model_checkpoint(checkpoint_to_load, model)
                validate_checkpoint_metadata_matches_config(
                    checkpoint_to_load,
                    checkpoint_metadata_for_validation(checkpoint_to_load, checkpoint_meta),
                    config,
                    require_config=bool(config.get("run_metadata_required", False)),
                )
                model.to(device)
                loaded_epoch = checkpoint_meta.get("epoch")
                suffix = f" at epoch {loaded_epoch}" if loaded_epoch is not None else ""
                print_master(f"Loaded best checkpoint {checkpoint_to_load}{suffix}", use_xla, xm, xr)
                final_test_checkpoint_source = checkpoint_to_load.as_posix()
            elif config["checkpoint"]:
                final_test_checkpoint_source = str(config["checkpoint"])
        elif args.eval_only and not config["checkpoint"]:
            print_master(
                "Eval-only requested without --checkpoint; evaluating current weights.",
                use_xla,
                xm,
                xr,
            )
        elif config["checkpoint"]:
            final_test_checkpoint_source = str(config["checkpoint"])
        elif train_loader is None and not config["checkpoint"]:
            print_master(
                "No train split is available and no checkpoint was provided; evaluating randomly initialized weights.",
                use_xla,
                xm,
                xr,
            )

        eval_epoch = int(config["epochs"]) if args.eval_only else 0
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, device, eval_epoch, config, args, use_xla, xm)
        best_acc = float(checkpoint_meta.get("best_acc") or val_acc)
        best_epoch = int(checkpoint_meta.get("best_epoch") or checkpoint_meta.get("epoch") or eval_epoch)
        if bool(config["save_csv"]) and is_master(use_xla, xm, xr):
            append_eval_metrics_csv(csv_path, eval_epoch, val_loss, val_acc, val_top5, best_acc, best_epoch)
        print_master(f"eval_loss={val_loss:.4f} eval_top1={val_acc:.2f} eval_top5={val_top5:.2f}", use_xla, xm, xr)
        if bool(config["final_test"]) and test_loader is not None:
            evaluate_final_test(
                model,
                test_loader,
                device,
                int(config["epochs"]),
                config,
                args,
                use_xla,
                xm,
                xr,
                csv_path,
                best_acc,
                best_epoch,
                checkpoint_source=final_test_checkpoint_source,
            )
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
    start_epoch = 1
    use_best_for_final_test = (
        bool(config["final_test"]) and str(config.get("final_test_checkpoint", "last")).lower() == "best"
    )
    best_model_state: dict[str, torch.Tensor] | None = None
    if config["checkpoint"]:
        start_epoch, best_acc, best_epoch, resumed_state = restore_training_state(
            checkpoint_meta,
            optimizer,
            scheduler,
            device,
        )
        if resumed_state:
            print_master(
                f"Resumed training state from {config['checkpoint']}; "
                f"start_epoch={start_epoch}; best_top1={best_acc:.2f}; best_epoch={best_epoch}",
                use_xla,
                xm,
                xr,
            )

    last_completed_epoch = max(start_epoch - 1, 0)
    last_checkpoint_epoch = 0
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, mixer, device, epoch, config, args, use_xla, xm, rank, world_size
        )
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, device, epoch, config, args, use_xla, xm)
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        if best_epoch == 0 or val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            if use_best_for_final_test:
                best_model_state = clone_model_state_dict(model)
            if bool(config["save_checkpoint"]) and is_master(use_xla, xm, xr):
                save_checkpoint(
                    checkpoint_root / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_acc,
                    config,
                    use_xla,
                    xm,
                    best_epoch=best_epoch,
                )
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
                best_epoch=best_epoch,
            )
        if bool(config["save_csv"]) and is_master(use_xla, xm, xr):
            train_error = pct_to_error(train_acc)
            val_error = pct_to_error(val_acc)
            val_top5_error = pct_to_error(val_top5)
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
                    "eval_top5_accuracy": f"{pct_to_fraction(val_top5):.6f}",
                    "eval_top5_error": f"{val_top5_error:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_top1": f"{val_acc:.6f}",
                    "val_top1_error": f"{val_error:.6f}",
                    "val_top5": f"{val_top5:.6f}",
                    "val_top5_error": f"{val_top5_error:.6f}",
                    "best_top1_error": f"{best_error:.6f}",
                    "best_epoch": best_epoch,
                    "best_top1": f"{best_acc:.6f}",
                },
            )

        print_master(
            f"epoch={epoch} train_loss={train_loss:.4f} train_top1={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_top1={val_acc:.2f} val_top5={val_top5:.2f} best_top1={best_acc:.2f}",
            use_xla,
            xm,
            xr,
        )
        last_completed_epoch = epoch
        if bool(config["save_checkpoint"]) and is_master(use_xla, xm, xr):
            save_checkpoint(
                checkpoint_root / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                config,
                use_xla,
                xm,
                best_epoch=best_epoch,
            )
            last_checkpoint_epoch = epoch

    if (
        bool(config["save_checkpoint"])
        and last_completed_epoch > 0
        and last_checkpoint_epoch != last_completed_epoch
        and is_master(use_xla, xm, xr)
    ):
        save_checkpoint(
            checkpoint_root / "last.pt",
            model,
            optimizer,
            scheduler,
            last_completed_epoch,
            best_acc,
            config,
            use_xla,
            xm,
            best_epoch=best_epoch,
        )
    if bool(config["final_test"]) and test_loader is not None:
        final_test_checkpoint_source = "current"
        if use_best_for_final_test:
            best_checkpoint_path = checkpoint_root / "best.pt"
            if best_model_state is None and best_checkpoint_path.exists():
                validate_checkpoint_metadata_matches_config(
                    best_checkpoint_path,
                    load_checkpoint_metadata_for_validation(best_checkpoint_path),
                    config,
                    require_config=bool(config.get("run_metadata_required", False)),
                )
            restore_source = restore_required_best_weights_for_final_test(
                model,
                best_model_state,
                best_checkpoint_path,
                device,
            )
            print_master(
                f"Final test using best validation weights from epoch={best_epoch} ({restore_source}).",
                use_xla,
                xm,
                xr,
            )
            final_test_checkpoint_source = restore_source
        evaluate_final_test(
            model,
            test_loader,
            device,
            int(config["epochs"]),
            config,
            args,
            use_xla,
            xm,
            xr,
            csv_path,
            best_acc,
            best_epoch,
            checkpoint_source=final_test_checkpoint_source,
        )
    print_master(f"Finished. best_top1={best_acc:.2f}", use_xla, xm, xr)


def main() -> None:
    args = parse_args()
    should_spawn = configure_xla_launch_environment(args)
    if should_spawn:
        launcher = _optional_xla_launcher()
        if launcher is None:
            raise RuntimeError("PyTorch/XLA is not installed. Cannot spawn XLA workers.")
        launcher(run_worker, args=(args,), start_method="spawn")
    else:
        run_worker(0, args)


if __name__ == "__main__":
    main()
