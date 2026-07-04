# AllTheMiXLA

Lightweight PyTorch/XLA FMix and MixUp reproduction code for CIFAR-10, CIFAR-100,
Tiny-ImageNet, and ImageNet-A evaluation.

The FMix operator follows the official Fourier-mask implementation from
`ecs-vlc/FMix`, and MixUp follows the standard batch-level Beta mixing recipe.
The default training presets follow the OpenMixup small-scale classification
benchmark settings.

## What Is Included

- Official-style FMix mask sampling in `allthemix/methods/fmix.py`.
- Standard batch-level MixUp in `allthemix/methods/mixup.py`.
- CIFAR-style PreAct-ResNet-18 and torchvision ResNet-101 backbones in `allthemix/networks/backbones`, with classifier heads in `allthemix/networks/heads`.
- CIFAR-10, CIFAR-100, original-layout Tiny-ImageNet, and ImageNet-A loaders in `allthemix/data`.
- A single CPU/CUDA/XLA trainer: `python -m allthemix.cli.train`.
- Two recipe families: `openmixup` and `official`.

## Layout

- `allthemix/data`: dataset loading and preprocessing pipelines.
- `allthemix/data/preprocessors`: sample-level basic augmentation and normalization.
- `allthemix/methods`: batch-level FMix and MixUp method implementations; baseline skips this stage.
- `allthemix/networks`: backbones, heads, classifiers, and model builder.
- `allthemix/training`: losses and training utilities.
- `allthemix/cli`: command-line training entry points and presets.
- `configs`: experiment configs grouped by dataset and backbone.
- `scripts/experiment_run`: batch launchers matching the config layout.
- `tests`: unit tests grouped by package area.

## Presets

| Dataset | Recipe | Epochs | Batch | LR | Scheduler | FMix alpha | MixUp alpha |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| CIFAR-10 | openmixup | 400 | 100 | 0.1 | cosine | 0.2 | 1.0 |
| CIFAR-100 | openmixup | 400 | 100 | 0.1 | cosine | 0.2 | 1.0 |
| Tiny-ImageNet | openmixup | 400 | 100 | 0.2 | cosine | 1.0 | 1.0 |
| CIFAR-10/100 | official | 200 | 128 | 0.1 | milestones 100,150 | 1.0 | 1.0 |
| Tiny-ImageNet | official | 200 | 128 | 0.1 | milestones 150,180 | 1.0 | 1.0 |
| ImageNet-A | official | eval only | 100 | 0.0 | n/a | n/a | n/a |

All recipes use `decay_power=3`, `max_soft=0`, SGD momentum `0.9`, and weight
decay `1e-4` unless overridden on the CLI.

## Training Flow

The training path is:

```text
data -> basic aug/preprocess -> batch -> optional FMix/MixUp method -> train loop
```

`basic aug` is sample-level augmentation inside the dataset transform:

- CIFAR: random crop with padding 4, random horizontal flip, tensor conversion, normalization.
- Tiny-ImageNet OpenMixup recipe: random resized crop 64 with bicubic interpolation, random horizontal flip, tensor conversion, ImageNet normalization.
- ImageNet-A eval: resize 256, center crop 224, tensor conversion, ImageNet normalization.
- Other eval/test: tensor conversion and normalization only.

FMix and MixUp are batch-level methods, so they run in the training loop after
DataLoader batching. Choose one with `method: fmix` or `method: mixup` in the
config, or use `method: baseline` to train with only basic augmentation.
FMix/MixUp configs enable `cross_device_shuffle: true`, so on XLA multi-process
runs the mixed partner batch is sampled from the gathered global batch rather
than only from each process-local mini-batch.

## Quick Checks

```powershell
python -m unittest discover -s tests
python -m allthemix.cli.train --help
```

A short CPU smoke run on CIFAR-10:

```powershell
python -m allthemix.cli.train `
  --config configs/cifar10/preact_resnet18/fmix.yaml `
  --download `
  --device cpu `
  --epochs 1 `
  --batch-size 8 `
  --num-workers 0 `
  --max-train-steps 2 `
  --max-val-steps 2
```

## XLA Runs

On a TPU VM, create and activate the local Python 3.10 `.venvxla` environment:

```bash
bash scripts/setup_tpu_venvxla.sh
source .venvxla/bin/activate
export PJRT_DEVICE=TPU
```

The setup script pins matching PyTorch/XLA wheels by default:
`torch==2.9.0`, `torchvision==0.24.0`, and `torch_xla[tpu]==2.9.0`.

If `.venvxla` already exists from another Python version, rebuild it:

```bash
RECREATE_VENV=1 bash scripts/setup_tpu_venvxla.sh
```

Verify TPU visibility:

```bash
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices()); print(xm.xla_real_devices())"
```

Then run:

```bash
python -m allthemix.cli.train \
  --config configs/cifar100/preact_resnet18/fmix.yaml \
  --download \
  --device xla \
  --num-cores 8 \
  --log-interval 0
```

For a quick TPU smoke test, add `--epochs 1 --max-train-steps 20 --max-val-steps 5`.

Run the three FMix PreAct-ResNet-18 configs:

```bash
bash scripts/experiment_run/run_cifar10_preact_resnet18.sh --download --device xla --num-cores 8
bash scripts/experiment_run/run_cifar100_preact_resnet18.sh --download --device xla --num-cores 8
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18.sh --device xla --num-cores 8
```

Run the matching MixUp PreAct-ResNet-18 configs:

```bash
bash scripts/experiment_run/run_cifar10_preact_resnet18_mixup.sh --download --device xla --num-cores 8
bash scripts/experiment_run/run_cifar100_preact_resnet18_mixup.sh --download --device xla --num-cores 8
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_mixup.sh --device xla --num-cores 8
```

Run the no-mix baseline configs:

```bash
bash scripts/experiment_run/run_cifar10_preact_resnet18_baseline.sh --download --device xla --num-cores 8
bash scripts/experiment_run/run_cifar100_preact_resnet18_baseline.sh --download --device xla --num-cores 8
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline.sh --device xla --num-cores 8
```

To mirror the older Tiny-ImageNet baseline config more closely
(`200 epochs`, `batch_size=128`, `lr=0.1`, `weight_decay=5e-4`,
`step` milestones `150,180`, `validation_split=0.1`, final test enabled), run:

```bash
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_legacy.sh --device xla --num-cores 8
```

Evaluate ImageNet-A with a checkpoint:

```bash
bash scripts/experiment_run/run_imagenet_a_torch_resnet101.sh \
  --checkpoint path/to/imagenet_resnet101_fmix.pt \
  --data-dir data/imagenet-a \
  --device xla \
  --num-cores 8
```

For an ImageNet ResNet-101 MixUp checkpoint, use:

```bash
bash scripts/experiment_run/run_imagenet_a_torch_resnet101_mixup.sh \
  --checkpoint path/to/imagenet_resnet101_mixup.pt \
  --data-dir data/imagenet-a \
  --device xla \
  --num-cores 8
```

ImageNet-A is eval-only and expects ImageFolder wnid class directories. A
1000-class ImageNet head is reduced to the official 200 ImageNet-A classes
before metrics are computed. The `torch_resnet101` config accepts plain
torchvision/official ResNet-101 state dict keys (`conv1`, `layer1`, `fc`, ...)
and maps them into the split backbone/head model.

Tiny-ImageNet expects either the original `tiny-imagenet-200` layout:

```text
data/tiny-imagenet-200/
  wnids.txt
  train/<wnid>/images/*.JPEG
  val/images/*.JPEG
  val/val_annotations.txt
```

or an ImageFolder-style `train/` and `val/` split.

ImageNet-A expects:

```text
data/imagenet-a/
  n01498041/*.jpg
  n01531178/*.jpg
  ...
```

## References

- Official FMix implementation: https://github.com/ecs-vlc/FMix
- Original MixUp CIFAR implementation: https://github.com/facebookresearch/mixup-cifar10
- OpenMixup CIFAR benchmark: https://github.com/Westlake-AI/openmixup/blob/main/docs/en/mixup_benchmarks/Mixup_cifar.md
- OpenMixup ImageNet/Tiny-ImageNet benchmark: https://github.com/Westlake-AI/openmixup/blob/main/docs/en/mixup_benchmarks/Mixup_imagenet.md
