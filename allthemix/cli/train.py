"""Train MixUp/FMIX PreAct-ResNet18 on CIFAR-10/100 or Tiny-ImageNet.

Use ``python -m allthemix.cli.train --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from allthemix.data import build_datasets
from allthemix.data.datasets import IMAGENET_A_INDICES_IN_1K, IMAGENET_A_NUM_CLASSES
from allthemix.methods import FMix, MixUp
from allthemix.networks import build_model
from allthemix.cli.presets import DATASETS, RECIPES, get_dataset_preset, get_recipe_preset, preset_dict
from allthemix.training.losses import fmix_cross_entropy, mixup_cross_entropy


def _optional_xla_import() -> dict[str, Any] | None:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ModuleNotFoundError:
        return None
    return {"xm": xm, "pl": pl, "xmp": xmp}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MixUp/FMIX PyTorch/XLA trainer")
    parser.add_argument("--config", default=None, help="YAML/JSON config path, e.g. configs/cifar10/preact_resnet18/fmix.yaml.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--recipe", choices=sorted(RECIPES), default=None)
    parser.add_argument("--method", choices=["eval", "fmix", "mixup", "none"], default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--download", action="store_true", default=None, help="Download CIFAR datasets if needed.")
    parser.add_argument("--no-augment", action="store_true", default=None, help="Disable train-time spatial augmentations.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=["cosine", "multistep"], default=None)
    parser.add_argument("--milestones", type=int, nargs="*", default=None)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--decay-power", type=float, default=None)
    parser.add_argument("--max-soft", type=float, default=None)
    parser.add_argument("--reformulate", action="store_true", default=None)
    parser.add_argument("--fmix-prob", type=float, default=None)
    parser.add_argument("--mix-prob", type=float, default=None, help="Batch-level mix method probability.")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xla"], default="auto")
    parser.add_argument("--num-cores", type=int, default=1, help="XLA processes to spawn when --device xla.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
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


def resolved_config(args: argparse.Namespace, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_config = raw_config or {}
    dataset_name = args.dataset or raw_config.get("dataset", "cifar10")
    recipe_name = args.recipe or raw_config.get("recipe", "openmixup")
    method_name = args.method or raw_config.get("method", "fmix")
    dataset = get_dataset_preset(dataset_name)
    recipe = get_recipe_preset(dataset_name, recipe_name)
    method_section = _section(raw_config, method_name)
    fmix_section = _section(raw_config, "fmix")

    config_augment = raw_config.get("use_basic_augmentation", raw_config.get("augment", True))
    use_basic_augmentation = bool(config_augment)
    if args.no_augment is True:
        use_basic_augmentation = False

    if args.mix_prob is not None:
        method_prob = args.mix_prob
    elif args.fmix_prob is not None:
        method_prob = args.fmix_prob
    else:
        method_prob = method_section.get("prob", raw_config.get("mix_prob", raw_config.get("fmix_prob", 1.0)))

    config = {
        "dataset": dataset_name,
        "recipe": recipe_name,
        "model": raw_config.get("model", "preact_resnet18"),
        "method": method_name,
        "data_dir": args.data_dir or raw_config.get("data_dir", "./data"),
        "output_dir": args.output_dir or raw_config.get("output_dir", "./runs/fmix"),
        "checkpoint": args.checkpoint or raw_config.get("checkpoint"),
        "download": bool(args.download if args.download is not None else raw_config.get("download", False)),
        "use_basic_augmentation": use_basic_augmentation,
        "num_classes": int(raw_config.get("num_classes", dataset.num_classes)),
        "image_size": dataset.image_size,
        "mean": dataset.mean,
        "std": dataset.std,
        "epochs": _choose(args, raw_config, "epochs", "training", "epochs", recipe.epochs),
        "batch_size": _choose(args, raw_config, "batch_size", "training", "batch_size", recipe.batch_size),
        "lr": _choose(args, raw_config, "lr", "training", "lr", recipe.lr),
        "momentum": _choose(args, raw_config, "momentum", "training", "momentum", recipe.momentum),
        "weight_decay": _choose(args, raw_config, "weight_decay", "training", "weight_decay", recipe.weight_decay),
        "scheduler": _choose(args, raw_config, "scheduler", "training", "scheduler", recipe.scheduler),
        "milestones": _choose(args, raw_config, "milestones", "training", "milestones", list(recipe.milestones)),
        "alpha": args.alpha if args.alpha is not None else method_section.get("alpha", fmix_section.get("alpha", recipe.alpha)),
        "decay_power": _choose(args, raw_config, "decay_power", "fmix", "decay_power", recipe.decay_power),
        "max_soft": _choose(args, raw_config, "max_soft", "fmix", "max_soft", recipe.max_soft),
        "transform_profile": recipe.transform_profile,
        "reformulate": _choose(args, raw_config, "reformulate", "fmix", "reformulate", False),
        "method_prob": method_prob,
        "fmix_prob": method_prob,
    }
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_master(use_xla: bool, xm: Any | None) -> bool:
    if not use_xla:
        return True
    return xm.is_master_ordinal()


def print_master(message: str, use_xla: bool, xm: Any | None) -> None:
    if use_xla:
        xm.master_print(message)
    else:
        print(message, flush=True)


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


def _xla_world_size(xm: Any) -> int:
    if hasattr(xm, "xrt_world_size"):
        return int(xm.xrt_world_size())
    if hasattr(xm, "world_size"):
        return int(xm.world_size())
    return 1


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
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]), eta_min=0.0)
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config["milestones"]), gamma=0.1)


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
    if method in {"none", "eval"}:
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
    raise ValueError(f"Unsupported mixed-sample method: {config['method']}")


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

    for step, (images, targets) in enumerate(loader, start=1):
        if args.max_train_steps is not None and step > args.max_train_steps:
            break
        if not use_xla:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if mixer is not None and config["method_prob"] > 0 and random.random() < config["method_prob"]:
            mixed = mixer(images, targets)
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
                if xm.is_master_ordinal():
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

    if use_xla:
        device = xm.xla_device()
        rank = xm.get_ordinal()
        world_size = _xla_world_size(xm)
    elif args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        rank = 0
        world_size = 1
    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1

    set_seed(args.seed + rank)
    raw_config = load_config(args.config)
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

    distributed = world_size > 1
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        if train_set is not None and distributed
        else None
    )
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

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

    if use_xla:
        if train_loader is not None:
            train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

    model = build_model(str(config["model"]), num_classes=int(config["num_classes"]))
    if config["checkpoint"]:
        checkpoint_meta = load_model_checkpoint(str(config["checkpoint"]), model)
        loaded_epoch = checkpoint_meta.get("epoch")
        suffix = f" at epoch {loaded_epoch}" if loaded_epoch is not None else ""
        print_master(f"Loaded checkpoint {config['checkpoint']}{suffix}", use_xla, xm)
    model = model.to(device)

    run_dir = Path(config["output_dir"]) / config["dataset"] / config["recipe"]
    if is_master(use_xla, xm):
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
    print_master(
        f"Starting {str(config['method']).upper()} {config['dataset']}/{config['recipe']} on {device}; "
        f"world_size={world_size}; config={json.dumps(config, sort_keys=True)}",
        use_xla,
        xm,
    )

    if train_loader is None or int(config["epochs"]) <= 0:
        if train_loader is None and not config["checkpoint"]:
            print_master(
                "No train split is available and no checkpoint was provided; evaluating randomly initialized weights.",
                use_xla,
                xm,
            )
        val_loss, val_acc = evaluate(model, val_loader, device, 0, config, args, use_xla, xm)
        print_master(f"eval_loss={val_loss:.4f} eval_top1={val_acc:.2f}", use_xla, xm)
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
    for epoch in range(1, int(config["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, mixer, device, epoch, config, args, use_xla, xm
        )
        val_loss, val_acc = evaluate(model, val_loader, device, epoch, config, args, use_xla, xm)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            if is_master(use_xla, xm):
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc, config, use_xla, xm)
        if args.save_every and epoch % args.save_every == 0 and is_master(use_xla, xm):
            save_checkpoint(
                run_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                config,
                use_xla,
                xm,
            )

        print_master(
            f"epoch={epoch} train_loss={train_loss:.4f} train_top1={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_top1={val_acc:.2f} best_top1={best_acc:.2f}",
            use_xla,
            xm,
        )

    if is_master(use_xla, xm):
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, int(config["epochs"]), best_acc, config, use_xla, xm)
    print_master(f"Finished. best_top1={best_acc:.2f}", use_xla, xm)


def main() -> None:
    args = parse_args()
    xla_modules = _optional_xla_import()
    should_spawn = args.device == "xla" and args.num_cores > 1
    if should_spawn:
        if xla_modules is None:
            raise RuntimeError("PyTorch/XLA is not installed. Cannot spawn XLA workers.")
        xla_modules["xmp"].spawn(run_worker, args=(args,), nprocs=args.num_cores)
    else:
        run_worker(0, args)


if __name__ == "__main__":
    main()
