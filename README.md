# AllTheMiXLA

Lightweight PyTorch/XLA FMix reproduction code for CIFAR-10, CIFAR-100, and
Tiny-ImageNet.

The FMix operator follows the official Fourier-mask implementation from
`ecs-vlc/FMix`. The default training presets follow the OpenMixup small-scale
classification benchmark settings.

## What Is Included

- Official-style FMix mask sampling in `allthemix/methods/fmix.py`.
- CIFAR-style PreAct-ResNet-18 split into `allthemix/networks/nn` and `allthemix/networks/heads`.
- CIFAR-10, CIFAR-100, and original-layout Tiny-ImageNet loaders in `allthemix/data`.
- A single CPU/CUDA/XLA trainer: `python -m allthemix.cli.train`.
- Two recipe families: `openmixup` and `official`.

## Layout

- `allthemix/data`: dataset loading and preprocessing pipelines.
- `allthemix/methods`: FMix method implementation.
- `allthemix/networks`: feature nets, heads, classifiers, and model builder.
- `allthemix/training`: losses and training utilities.
- `allthemix/cli`: command-line training entry points and presets.
- `configs`: experiment configs grouped by dataset and backbone.
- `scripts/experiment_run`: batch launchers matching the config layout.
- `tests`: unit tests grouped by package area.

## Presets

| Dataset | Recipe | Epochs | Batch | LR | Scheduler | FMix alpha |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| CIFAR-10 | openmixup | 400 | 100 | 0.1 | cosine | 0.2 |
| CIFAR-100 | openmixup | 400 | 100 | 0.1 | cosine | 0.2 |
| Tiny-ImageNet | openmixup | 400 | 100 | 0.2 | cosine | 1.0 |
| CIFAR-10/100 | official | 200 | 128 | 0.1 | milestones 100,150 | 1.0 |
| Tiny-ImageNet | official | 200 | 128 | 0.1 | milestones 150,180 | 1.0 |

All recipes use `decay_power=3`, `max_soft=0`, SGD momentum `0.9`, and weight
decay `1e-4` unless overridden on the CLI.

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

Install a `torch_xla` build that matches your PyTorch/XLA runtime, then run:

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

Tiny-ImageNet expects either the original `tiny-imagenet-200` layout:

```text
data/tiny-imagenet-200/
  wnids.txt
  train/<wnid>/images/*.JPEG
  val/images/*.JPEG
  val/val_annotations.txt
```

or an ImageFolder-style `train/` and `val/` split.

## References

- Official FMix implementation: https://github.com/ecs-vlc/FMix
- OpenMixup CIFAR benchmark: https://github.com/Westlake-AI/openmixup/blob/main/docs/en/mixup_benchmarks/Mixup_cifar.md
- OpenMixup ImageNet/Tiny-ImageNet benchmark: https://github.com/Westlake-AI/openmixup/blob/main/docs/en/mixup_benchmarks/Mixup_imagenet.md
