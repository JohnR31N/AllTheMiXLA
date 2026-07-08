"""Build cached train saliency maps for SaliencyMix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from allthemix.cli.presets import (
    DATASET_EXPECTED_SPLIT_COUNTS,
    get_dataset_preset,
    get_recipe_preset,
    normalize_dataset_name,
)
from allthemix.cli.train import (
    data_loader_seed_kwargs,
    load_config,
    parse_nonnegative_int_arg,
    parse_positive_int_arg,
    relocate_relative_saliency_path,
    parse_seed_arg,
    resolve_saliency_storage_paths,
    set_seed,
    validate_seed,
)
from allthemix.data.pipeline import build_datasets
from allthemix.data.saliency_dataset import saliency_array_is_finite, saliency_path_candidates
from allthemix.methods.saliencymix import compute_gradient_saliency_maps
from allthemix.methods.guided_sr import compute_spectral_residual_saliency_maps, minmax_normalize_saliency_maps


SaliencyFn = Callable[[torch.Tensor], torch.Tensor]
OPENCV_METHODS = {"opencv", "opencv_finegrained", "finegrained"}
GRADIENT_METHODS = {"gradient", "grad"}
SPECTRAL_RESIDUAL_METHODS = {"spectral_residual", "sr", "guided_sr", "online"}
COLUMN_STRIPE_STD_RATIO = 2.5
CACHE_BUILDER_VERSION = 4
EXPECTED_CACHE_TRAIN_EXAMPLES = {
    dataset_name: counts["train"]
    for dataset_name, counts in DATASET_EXPECTED_SPLIT_COUNTS.items()
    if "train" in counts
}


def _device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def parse_blur_kernel_arg(value: str) -> int:
    try:
        kernel = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--blur-kernel must be an integer.") from exc
    if kernel <= 0 or kernel % 2 == 0:
        raise argparse.ArgumentTypeError("--blur-kernel must be a positive odd integer.")
    return kernel


def blur_kernel_from_config(raw_config: dict[str, object], override: int | None) -> int:
    if override is not None:
        return int(override)
    value = raw_config.get("guidedmixup_blur_kernel", raw_config.get("blur_kernel", 7))
    try:
        kernel = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"guidedmixup_blur_kernel must be an integer, got {value!r}.") from exc
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"guidedmixup_blur_kernel must be a positive odd integer, got {value!r}.")
    return kernel


def validate_saliency_cache_dataset_length(dataset_name: str, dataset) -> None:
    normalized_name = normalize_dataset_name(dataset_name)
    expected_count = EXPECTED_CACHE_TRAIN_EXAMPLES.get(normalized_name)
    if expected_count is None:
        return
    try:
        actual_count = len(dataset)
    except TypeError as exc:
        raise ValueError(f"Cannot validate {normalized_name} saliency cache dataset length.") from exc
    if int(actual_count) != int(expected_count):
        raise ValueError(
            f"{normalized_name} saliency cache requires the complete train split "
            f"({expected_count} examples), got {actual_count}. "
            "Fix/download the dataset before building formal saliency caches."
        )


def _normalize_numpy_saliency_map(saliency_map: np.ndarray) -> np.ndarray:
    saliency_map = saliency_map.astype(np.float32)
    saliency_map = saliency_map - float(np.min(saliency_map))
    max_value = float(np.max(saliency_map))
    if max_value < 1e-8:
        return np.zeros_like(saliency_map, dtype=np.float32)
    return (saliency_map / max_value).astype(np.float32)


def _numpy_gradient_saliency_map(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if float(image.max()) > 1.0:
        image = image / 255.0
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    dx = np.zeros_like(gray, dtype=np.float32)
    dy = np.zeros_like(gray, dtype=np.float32)
    dx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    dy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return _normalize_numpy_saliency_map(dx + dy)


def _is_suspicious_saliency_map(saliency_map: np.ndarray) -> bool:
    if saliency_map.ndim != 2 or not np.all(np.isfinite(saliency_map)):
        return True
    if float(np.max(saliency_map) - np.min(saliency_map)) < 1e-8:
        return True
    row_mean_std = float(np.std(np.mean(saliency_map, axis=1)))
    col_mean_std = float(np.std(np.mean(saliency_map, axis=0)))
    return col_mean_std > row_mean_std * COLUMN_STRIPE_STD_RATIO


def _opencv_finegrained_saliency_map(image: np.ndarray, allow_gradient_fallback: bool = False) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError:
        if allow_gradient_fallback:
            return _numpy_gradient_saliency_map(image)
        raise RuntimeError(
            "OpenCV saliency backend is not installed. Install opencv-contrib-python-headless "
            "or use --method gradient for a non-table smoke/debug cache."
        ) from None

    if not hasattr(cv2, "saliency") or not hasattr(cv2.saliency, "StaticSaliencyFineGrained_create"):
        if allow_gradient_fallback:
            return _numpy_gradient_saliency_map(image)
        raise RuntimeError(
            "OpenCV was imported but cv2.saliency.StaticSaliencyFineGrained_create is unavailable. "
            "Install opencv-contrib-python-headless or use --method gradient for smoke/debug."
        )

    image_uint8 = image.astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    detector = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = detector.computeSaliency(image_bgr)
    if not success:
        if allow_gradient_fallback:
            return _numpy_gradient_saliency_map(image)
        raise RuntimeError("OpenCV fine-grained saliency failed; use --allow-gradient-fallback only for smoke/debug.")

    saliency_map = _normalize_numpy_saliency_map(saliency_map)
    if _is_suspicious_saliency_map(saliency_map):
        if allow_gradient_fallback:
            return _numpy_gradient_saliency_map(image)
        raise RuntimeError("OpenCV saliency map looked invalid; use --allow-gradient-fallback only for smoke/debug.")
    return saliency_map.astype(np.float32)


def _tensor_to_uint8_images(
    images: torch.Tensor,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
) -> np.ndarray:
    images = _tensor_to_unit_images(images, mean=mean, std=std).detach().cpu()
    images = images.permute(0, 2, 3, 1).numpy()
    return (images * 255.0).round().astype(np.uint8)


def _tensor_to_unit_images(
    images: torch.Tensor,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
) -> torch.Tensor:
    images = images.detach().float()
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
        images = images * std_tensor + mean_tensor
    return images.clamp(0.0, 1.0)


def _normalize_tensor_saliency_maps(saliency_maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return minmax_normalize_saliency_maps(saliency_maps, eps=eps)


def _compute_opencv_maps(
    images: torch.Tensor,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    allow_gradient_fallback: bool = False,
) -> torch.Tensor:
    uint8_images = _tensor_to_uint8_images(images, mean=mean, std=std)
    maps = [
        _opencv_finegrained_saliency_map(image, allow_gradient_fallback=allow_gradient_fallback)
        for image in uint8_images
    ]
    maps_np = np.stack(maps, axis=0).astype(np.float32)
    return torch.from_numpy(maps_np[:, None]).to(device=images.device)


def _compute_maps(
    images: torch.Tensor,
    method: str,
    blur_kernel: int,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    allow_gradient_fallback: bool = False,
) -> torch.Tensor:
    method = str(method).lower()
    if method in OPENCV_METHODS:
        return _compute_opencv_maps(
            images,
            mean=mean,
            std=std,
            allow_gradient_fallback=allow_gradient_fallback,
        )
    unit_images = _tensor_to_unit_images(images, mean=mean, std=std)
    if method in GRADIENT_METHODS:
        return _normalize_tensor_saliency_maps(compute_gradient_saliency_maps(unit_images))
    if method in SPECTRAL_RESIDUAL_METHODS:
        return _normalize_tensor_saliency_maps(
            compute_spectral_residual_saliency_maps(unit_images, blur_kernel=int(blur_kernel))
        )
    raise ValueError(f"Unsupported saliency cache method: {method}")


def _unpack_images(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _metadata_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, (tuple, list)):
        return list(actual) == list(expected) if isinstance(actual, (tuple, list)) else False
    return actual == expected


def _close_numpy_mmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _temporary_cache_path(path: Path) -> Path:
    if path.suffix == ".npy":
        return path.with_name(f".{path.stem}.tmp{path.suffix}")
    return path.with_name(f".{path.name}.tmp")


def _backup_cache_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.bak")


def _recover_cache_backup_if_needed(
    output_path: Path,
    metadata_path: Path,
    backup_data_path: Path,
    backup_metadata_path: Path,
) -> None:
    data_backup_exists = backup_data_path.exists()
    metadata_backup_exists = backup_metadata_path.exists()
    if not data_backup_exists and not metadata_backup_exists:
        return
    final_pair_complete = output_path.exists() and metadata_path.exists()
    backup_pair_complete = data_backup_exists and metadata_backup_exists
    if backup_pair_complete and not final_pair_complete:
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        backup_data_path.replace(output_path)
        backup_metadata_path.replace(metadata_path)
        return
    if data_backup_exists and not output_path.exists():
        backup_data_path.replace(output_path)
    if metadata_backup_exists and not metadata_path.exists():
        backup_metadata_path.replace(metadata_path)
    if final_pair_complete:
        backup_data_path.unlink(missing_ok=True)
        backup_metadata_path.unlink(missing_ok=True)


def _save_cache_atomic(output_path: Path, saliency_array: np.ndarray, metadata: dict[str, object]) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    temp_data_path = _temporary_cache_path(output_path)
    temp_metadata_path = _temporary_cache_path(metadata_path)
    backup_data_path = _backup_cache_path(output_path)
    backup_metadata_path = _backup_cache_path(metadata_path)
    _recover_cache_backup_if_needed(output_path, metadata_path, backup_data_path, backup_metadata_path)
    for temp_path in (temp_data_path, temp_metadata_path, backup_data_path, backup_metadata_path):
        temp_path.unlink(missing_ok=True)
    data_backed_up = False
    metadata_backed_up = False
    data_installed = False
    metadata_installed = False
    try:
        with temp_data_path.open("wb") as handle:
            np.save(handle, saliency_array)
        temp_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if output_path.exists():
            output_path.replace(backup_data_path)
            data_backed_up = True
        if metadata_path.exists():
            metadata_path.replace(backup_metadata_path)
            metadata_backed_up = True
        temp_data_path.replace(output_path)
        data_installed = True
        temp_metadata_path.replace(metadata_path)
        metadata_installed = True
        backup_data_path.unlink(missing_ok=True)
        backup_metadata_path.unlink(missing_ok=True)
        data_backed_up = False
        metadata_backed_up = False
    except Exception:
        if data_installed:
            output_path.unlink(missing_ok=True)
        if metadata_installed:
            metadata_path.unlink(missing_ok=True)
        if data_backed_up and backup_data_path.exists():
            backup_data_path.replace(output_path)
            data_backed_up = False
        if metadata_backed_up and backup_metadata_path.exists():
            backup_metadata_path.replace(metadata_path)
            metadata_backed_up = False
        for temp_path in (temp_data_path, temp_metadata_path):
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        if not data_backed_up:
            backup_data_path.unlink(missing_ok=True)
        if not metadata_backed_up:
            backup_metadata_path.unlink(missing_ok=True)


def _existing_cache_matches_request(
    output_path: Path,
    method: str,
    dtype: str,
    metadata_context: dict[str, object] | None,
    allow_gradient_fallback: bool = False,
    expected_count: int | None = None,
) -> bool:
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    try:
        if int(metadata.get("builder_version", 0)) < CACHE_BUILDER_VERSION:
            return False
    except (TypeError, ValueError):
        return False
    if str(metadata.get("method", "")).lower() != str(method).lower():
        return False
    if str(method).lower() in SPECTRAL_RESIDUAL_METHODS:
        try:
            expected_blur_kernel = int((metadata_context or {}).get("blur_kernel", 7))
            actual_blur_kernel = int(metadata.get("blur_kernel"))
        except (TypeError, ValueError):
            return False
        if actual_blur_kernel != expected_blur_kernel:
            return False
    if "allow_gradient_fallback" not in metadata:
        return False
    if bool(metadata.get("allow_gradient_fallback")) and not bool(allow_gradient_fallback):
        return False
    expected_dtype = "float16" if str(dtype).lower() == "float16" else "float32"
    if str(metadata.get("dtype", "")).lower() != expected_dtype:
        return False
    try:
        saliency_maps = np.load(output_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    try:
        if str(saliency_maps.dtype).lower() != expected_dtype:
            return False
        try:
            metadata_count = int(metadata.get("count"))
            metadata_shape = tuple(int(value) for value in metadata.get("shape"))
        except (TypeError, ValueError):
            return False
        if expected_count is not None and metadata_count != int(expected_count):
            return False
        if metadata_count != int(saliency_maps.shape[0]):
            return False
        if metadata_shape != tuple(int(value) for value in saliency_maps.shape):
            return False
        if not saliency_array_is_finite(saliency_maps):
            return False
        for key, expected in (metadata_context or {}).items():
            if key not in metadata or not _metadata_value_matches(metadata.get(key), expected):
                return False
        return bool(metadata.get("raw_unit_images")) and bool(metadata.get("minmax_normalized"))
    finally:
        _close_numpy_mmap(saliency_maps)


def build_saliency_maps_from_dataset(
    dataset,
    output_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int | None = 0,
    device: torch.device | str = "cpu",
    method: str = "opencv",
    blur_kernel: int = 7,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    dtype: str = "float32",
    log_interval: int = 50,
    saliency_fn: SaliencyFn | None = None,
    metadata_context: dict[str, object] | None = None,
    allow_gradient_fallback: bool = False,
) -> Path:
    """Compute and save saliency maps in dataset iteration order."""

    output_path = Path(output_path)
    batch_size = int(batch_size)
    num_workers = int(num_workers)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}.")
    try:
        dataset_count = int(len(dataset))
    except TypeError:
        dataset_count = None
    expected_count = None
    if dataset_count is not None:
        expected_count = min(dataset_count, limit) if limit is not None else dataset_count
    if output_path.exists() and not overwrite:
        if _existing_cache_matches_request(
            output_path,
            method,
            dtype,
            metadata_context,
            allow_gradient_fallback=allow_gradient_fallback,
            expected_count=expected_count,
        ):
            return output_path
        raise FileExistsError(
            f"Existing saliency cache at {output_path} is stale or incompatible with this request. "
            "Re-run with --overwrite to rebuild it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    np_dtype = np.float16 if str(dtype).lower() == "float16" else np.float32
    compute_fn = saliency_fn or (
        lambda images: _compute_maps(
            images,
            method=method,
            blur_kernel=blur_kernel,
            mean=mean,
            std=std,
            allow_gradient_fallback=allow_gradient_fallback,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        **data_loader_seed_kwargs(seed, offset=30_000),
    )

    maps: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            images = _unpack_images(batch).to(device=device, dtype=torch.float32, non_blocking=True)
            saliency_maps = compute_fn(images)
            if saliency_maps.dim() == 4:
                saliency_maps = saliency_maps[:, 0]
            if saliency_maps.dim() != 3:
                raise ValueError(f"saliency maps must have shape N,H,W or N,1,H,W, got {tuple(saliency_maps.shape)}")

            batch_maps = saliency_maps.detach().cpu().numpy().astype(np_dtype, copy=False)
            if limit is not None:
                remaining = max(limit - seen, 0)
                batch_maps = batch_maps[:remaining]
            maps.append(batch_maps)
            seen += int(batch_maps.shape[0])

            if log_interval and step % int(log_interval) == 0:
                print(f"built_saliency_maps={seen}", flush=True)
            if limit is not None and seen >= limit:
                break

    if not maps:
        raise ValueError("No saliency maps were generated; dataset appears empty.")
    saliency_array = np.concatenate(maps, axis=0)
    if int(saliency_array.shape[0]) <= 0:
        raise ValueError("No saliency maps were generated; dataset appears empty or limit is zero.")
    if not np.isfinite(saliency_array).all():
        raise ValueError("Generated saliency maps contain NaN or infinite values; refusing to write cache.")
    metadata = {
        **(metadata_context or {}),
        "builder": "allthemix.cli.build_saliency_cache",
        "builder_version": CACHE_BUILDER_VERSION,
        "method": str(method),
        "blur_kernel": int(blur_kernel),
        "count": int(saliency_array.shape[0]),
        "shape": list(saliency_array.shape),
        "dtype": str(saliency_array.dtype),
        "raw_unit_images": True,
        "minmax_normalized": True,
        "allow_gradient_fallback": bool(allow_gradient_fallback),
    }
    _save_cache_atomic(output_path, saliency_array, metadata)
    print(f"Saved saliency maps to: {output_path}")
    print(f"Shape: {saliency_array.shape}; dtype={saliency_array.dtype}")
    print(f"Min: {saliency_array.min():.6f}; max={saliency_array.max():.6f}; mean={saliency_array.mean():.6f}")
    return output_path


def output_path_from_args(args: argparse.Namespace, raw_config: dict) -> Path:
    if args.output:
        return Path(relocate_relative_saliency_path(args.output, args.saliency_dir) or args.output)
    saliency_dir, saliency_path = resolve_saliency_storage_paths(
        raw_config,
        data_dir_override=args.data_dir,
        saliency_dir_override=args.saliency_dir,
    )
    if saliency_path:
        return Path(saliency_path)
    dataset_for_filename = str(raw_config.get("dataset") or args.dataset)
    return saliency_path_candidates(dataset_for_filename, saliency_dir)[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached train saliency maps for SaliencyMix.")
    parser.add_argument("--config", default="configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--recipe", default=None, choices=["official", "openmixup"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--saliency-dir", default=None)
    parser.add_argument("--output", "--saliency-path", dest="output", default=None)
    parser.add_argument(
        "--method",
        choices=["opencv", "opencv_finegrained", "finegrained", "spectral_residual", "sr", "guided_sr", "gradient", "grad"],
        default="opencv",
    )
    parser.add_argument("--blur-kernel", type=parse_blur_kernel_arg, default=None)
    parser.add_argument("--batch-size", type=parse_positive_int_arg, default=128)
    parser.add_argument("--num-workers", type=parse_nonnegative_int_arg, default=4)
    parser.add_argument("--seed", type=parse_seed_arg, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--limit", type=parse_positive_int_arg, default=None, help="Optional number of train examples for cache smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-interval", type=parse_nonnegative_int_arg, default=50)
    parser.add_argument(
        "--allow-gradient-fallback",
        action="store_true",
        help="Allow OpenCV saliency failures to fall back to gradient maps. For smoke/debug only; table preflight rejects these caches.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.seed = validate_seed(args.seed)
    set_seed(args.seed)
    raw_config = load_config(args.config)
    dataset_name = normalize_dataset_name(args.dataset or raw_config.get("dataset", "tiny_imagenet"))
    recipe_name = args.recipe or raw_config.get("recipe", "openmixup")
    data_dir = args.data_dir or raw_config.get("data_dir", "./data")
    blur_kernel = blur_kernel_from_config(raw_config, args.blur_kernel)
    preset = get_dataset_preset(dataset_name)
    recipe = get_recipe_preset(dataset_name, recipe_name)

    train_set, _ = build_datasets(
        preset,
        recipe.transform_profile,
        data_dir=data_dir,
        download=False,
        use_basic_augmentation=False,
        augmentation_recipe="none",
    )
    if train_set is None:
        raise ValueError(f"Dataset {dataset_name} does not provide a training split.")
    validate_saliency_cache_dataset_length(dataset_name, train_set)

    output_path = output_path_from_args(args, raw_config)
    build_saliency_maps_from_dataset(
        train_set,
        output_path=output_path,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        device=_device_from_arg(str(args.device)),
        method=str(args.method),
        blur_kernel=int(blur_kernel),
        mean=tuple(preset.mean),
        std=tuple(preset.std),
        overwrite=bool(args.overwrite),
        limit=args.limit,
        dtype=str(args.dtype),
        log_interval=int(args.log_interval),
        allow_gradient_fallback=bool(args.allow_gradient_fallback),
        metadata_context={
            "dataset": dataset_name,
            "recipe": recipe_name,
            "transform_profile": recipe.transform_profile,
            "image_size": int(preset.image_size),
            "mean": list(preset.mean),
            "std": list(preset.std),
            "base_transform": "tensor_normalize_only",
            "blur_kernel": int(blur_kernel),
        },
    )


if __name__ == "__main__":
    main()
