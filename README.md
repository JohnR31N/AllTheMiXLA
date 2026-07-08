# AllTheMiXLA

Lightweight PyTorch/XLA MixDA reproduction code for CIFAR-10, CIFAR-100,
Tiny-ImageNet, and ImageNet-A evaluation.

The FMix operator follows the official Fourier-mask implementation from
`ecs-vlc/FMix`, and MixUp follows the standard batch-level Beta mixing recipe.
The default training presets follow the OpenMixup small-scale classification
benchmark settings.

## What Is Included

- Official-style FMix mask sampling in `allthemix/methods/fmix.py`.
- Batch-level MixUp, CutMix, ResizeMix, SaliencyMix, Guided-SR, and CatchUpMix in `allthemix/methods`.
- CIFAR-style PreAct-ResNet-18 and torchvision ResNet-101 backbones in `allthemix/networks/backbones`, with classifier heads in `allthemix/networks/heads`.
- CIFAR-10, CIFAR-100, original or class-folder Tiny-ImageNet, and ImageNet-A loaders in `allthemix/data`.
- A single CPU/CUDA/XLA trainer: `python -m allthemix.cli.train`.
- Two recipe families: `openmixup` and `official`.

## Layout

- `allthemix/data`: dataset loading and preprocessing pipelines.
- `allthemix/data/preprocessors`: sample-level basic augmentation and normalization.
- `allthemix/methods`: batch-level MixDA method implementations; baseline skips this stage.
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
data -> basic aug/preprocess -> batch -> optional MixDA method -> train loop
```

`basic aug` is sample-level augmentation inside the dataset transform:

- CIFAR: random crop with padding 4, random horizontal flip, tensor conversion, normalization.
- Tiny-ImageNet OpenMixup recipe: random resized crop 64 with bicubic interpolation, random horizontal flip, tensor conversion, ImageNet normalization.
- ImageNet-A eval: resize 256, center crop 224, tensor conversion, ImageNet normalization.
- Other eval/test: tensor conversion and normalization only.

Configs may also set an explicit `aug_recipe`. The Tiny-ImageNet table configs
use `basic_aug: false` with `aug_recipe: tiny_openmixup`, matching the
AllTheMix xla4 configs while producing the same OpenMixup-style random resized
crop and horizontal flip.

MixDA methods run in the training loop after DataLoader batching. Choose one
with `method: fmix`, `method: mixup`, `method: cutmix`, `method: resizemix`,
`method: saliencymix`, `method: guided_sr`, or `method: catchupmix` in the
config, or use `method: baseline` to train with only basic augmentation.
AllTheMix-style `method: guidedmixup` configs are accepted as the same
Guided-SR/GuidedMixup implementation.
Most Tiny-ImageNet XLA configs enable `cross_device_shuffle: true`, so on
multi-process TPU runs the mixed partner batch is sampled from the gathered
global batch rather than only from each process-local mini-batch. CatchUpMix
keeps feature mixing process-local because it mixes intermediate feature
channels inside the backbone.

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

## Tiny-ImageNet Data

Tiny-ImageNet is not downloaded by `--download`. To only download or reuse the
dataset under `./data/tiny-imagenet-200`, run:

```bash
bash scripts/download_tiny_imagenet.sh --data-dir ./data
```

After creating and activating `.venvxla`, the prepare script can do both steps:
download or reuse Tiny-ImageNet, then run the strict Tiny-ImageNet XLA4
preflight. Pass downloader options before `--`, and extra preflight options
after it, for example:

```bash
MIN_FREE_GB=20 bash scripts/experiment_run/prepare_tiny_imagenet_xla4.sh \
  --data-dir /mnt/tiny \
  -- \
  --train-arg=--saliency-dir --train-arg=/mnt/cache
```

The downloader uses the CS231n `tiny-imagenet-200.zip` archive by default,
checks the known MD5 when `md5sum` is available, and reuses an existing
`tiny-imagenet-200/train` plus `tiny-imagenet-200/val` layout only when the
expected image counts are present (`100000` train and `10000` validation
images). Pass `--force` to re-extract anyway. If you intentionally use a trusted
mirror or a manually copied archive, pass `--md5 <hash>` or `--skip-md5`.
When Tiny-ImageNet is missing, `summarize --format preflight` prints the
matching `download_tiny_imagenet.sh --data-dir ...` command.

## XLA Runs

On a TPU VM, create and activate the local Python 3.10 `.venvxla` environment:

```bash
bash scripts/setup_tpu_venvxla.sh
source .venvxla/bin/activate
export PJRT_DEVICE=TPU
export PYTHONNOUSERSITE=1
```

The setup script pins matching PyTorch/XLA wheels by default:
`torch==2.9.0`, `torchvision==0.24.0`, and `torch_xla[tpu]==2.9.0`.

If `.venvxla` already exists from another Python version, rebuild it:

```bash
RECREATE_VENV=1 bash scripts/setup_tpu_venvxla.sh
```

Verify TPU visibility:

```bash
python -m allthemix.cli.verify_xla_env --skip-device-check
PJRT_DEVICE=TPU python -m allthemix.cli.verify_xla_env --require-tpu --expected-tpu-devices 4 --require-venv-name .venvxla
```

The verifier also checks `opencv_saliency`; this must be `ok` for formal
SaliencyMix cache builds. Use `--skip-opencv-check` only for non-SaliencyMix
debug environments.
When `--require-venv-name .venvxla` is used, the verifier also requires
`torch`, `torchvision`, and `torch_xla` to be imported from inside that active
virtualenv, catching stale `~/.local` wheels before they can crash a TPU run.

The strict Tiny-ImageNet TPU preflight wraps the same environment, protocol,
storage-root, data, cache, and disk-space gates before a long run:

```bash
bash scripts/experiment_run/preflight_tiny_imagenet_xla4.sh
```

The script defaults `PJRT_DEVICE=TPU` for this TPU-only check.
Set `MIN_FREE_GB=20` to require more free space. Additional summary args are
forwarded, for example `--train-arg=--data-dir --train-arg=/mnt/tiny`.
Without `--check-env`, preflight stays filesystem/config-only so it can still be
used from a local machine that does not have PyTorch/XLA installed.
When using external mounts for data, saliency caches, outputs, or checkpoints,
add `--require-existing-storage-roots` to catch a missing mount directory before
the run creates files on the root disk. Repo-local `outputs/` and
`checkpoints/` are still treated as creatable fresh-run directories.

For XLA launches, `--num-cores N` sets `TPU_NUM_DEVICES=N` before importing the
launcher, so stale shell values from an older tmux session do not change either
the xla4 table run size or a single-core debug launch.

Then run:

```bash
python -m allthemix.cli.train \
  --config configs/cifar100/preact_resnet18/fmix.yaml \
  --download \
  --device xla \
  --num-cores 4 \
  --log-interval 0
```

For a quick TPU smoke test, add `--epochs 1 --max-train-steps 20 --max-val-steps 5`.

Run the three FMix PreAct-ResNet-18 configs:

```bash
bash scripts/experiment_run/run_cifar10_preact_resnet18.sh --download --device xla --num-cores 4
bash scripts/experiment_run/run_cifar100_preact_resnet18.sh --download --device xla --num-cores 4
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18.sh --device xla --num-cores 4
```

Run the matching MixUp PreAct-ResNet-18 configs:

```bash
bash scripts/experiment_run/run_cifar10_preact_resnet18_mixup.sh --download --device xla --num-cores 4
bash scripts/experiment_run/run_cifar100_preact_resnet18_mixup.sh --download --device xla --num-cores 4
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_mixup.sh --device xla --num-cores 4
```

Run the no-mix baseline configs:

```bash
bash scripts/experiment_run/run_cifar10_preact_resnet18_baseline.sh --download --device xla --num-cores 4
bash scripts/experiment_run/run_cifar100_preact_resnet18_baseline.sh --download --device xla --num-cores 4
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline.sh --device xla --num-cores 4
```

The non-`xla4` Tiny-ImageNet scripts are kept for legacy compatibility. For the
Tiny-ImageNet table runs, use the `*_xla4` configs. They mirror the older
AllTheMix-style split protocol with per-device batch `32` (global batch `128`
on 4 TPU devices), `200 epochs`, `lr=0.1`, `weight_decay=5e-4`, `step`
milestones `150,180`, `validation_split=0.1`, and final test enabled with
`final_test_checkpoint: best`, so the Tiny-ImageNet test split is evaluated
with the best validation checkpoint rather than the last epoch weights. The
sample-level transform is OpenMixup Tiny-ImageNet style
`RandomResizedCrop(64, bicubic) + RandomHorizontalFlip`: non-saliency methods
use `aug_recipe: tiny_openmixup`, while SaliencyMix and Guided-SR disable the
base image augmentation and use paired `sal_aug_recipe: tiny_openmixup` so image
and saliency maps stay aligned. XLA cross-device shuffle follows the AllTheMix
Tiny xla4 distributed path: MixUp, FMix, ResizeMix, and SaliencyMix use a global
random permutation, while CutMix keeps its configured no-repeat partners.
Guided-SR follows the GuidedMixup SR-style greedy pairing setting, so
`cross_device_shuffle` is disabled for that config. ResizeMix
explicitly uses the AllTheMix Tiny xla4 uniform `scope=(0.1, 0.4)` sampling
(`resizemix_use_alpha: false`); OpenMixup's public Tiny-ImageNet configs also
support a `use_alpha: true` beta-sampled size variant, which this trainer can
enable via `resizemix_use_alpha: true` for separate ablations. The trainer
records `global_batch_size: 128` in run
metadata and, on XLA launches, checks that `batch_size * world_size` matches it
before training starts.
The `run_tiny_imagenet_preact_resnet18_*_xla4.sh` entrypoints are TPU-first:
they default `PJRT_DEVICE=TPU`, `--device xla`, `--num-cores ${NUM_CORES:-4}`,
and `--num-workers ${NUM_WORKERS:-0}`. Explicit command-line arguments still
win because they are appended after those defaults. They also require an active
virtualenv named `.venvxla` by default, so a post-reboot shell fails early with
`source .venvxla/bin/activate` instructions instead of importing stale user-site
`torch_xla` wheels.

This table protocol is intentionally separate from the public OpenMixup
Tiny-ImageNet benchmark, which uses 400 epochs, global batch 100, `lr=0.2`,
cosine scheduling, and reports the last-10-epoch median accuracy. The summary
JSON and suite manifest therefore record
`protocol_id=allthemix_split200_openmixup_aug_bestval_xla4`; use
`--metric last10_median` only when you want an OpenMixup-style reporting view
of the same local 200-epoch table runs.

```bash
python -m allthemix.cli.summarize --preset tiny-imagenet-xla4 --format protocol --require-complete
python -m allthemix.cli.summarize --preset tiny-imagenet-xla4 --format preflight --require-complete
```

The protocol check validates both raw YAML fields and the resolved trainer
config, so accidental changes such as switching the table configs away from the
OpenMixup recipe are caught before a long TPU run starts. The preflight check
adds launch-time filesystem checks: Tiny-ImageNet data must be discoverable and
complete (`100000` train images and `10000` validation images), while a missing
SaliencyMix cache is reported as `will_build` because the suite can generate it
before training. Guided-SR uses online spectral-residual saliency in the formal
table protocol and does not need a cache; the online saliency path denormalizes
the augmented model tensor back to unit image space before computing per-image
min-max normalized SR maps.
When `--train-arg=--eval-only` is
used, preflight also reports which incomplete rows can refresh from
`best.pt` and which rows will fall back to full training. Suite `--train-arg`
path overrides such as `--data-dir` are passed through to preflight, cache
building, and training.

Run the whole missing-run suite with protocol check, cached-saliency handling,
completion check, and final LaTeX table output. The suite prints both the
default summary (best-validation-checkpoint `final_test`) and the
OpenMixup-style `last10_median` summary, and writes stable table artifacts to
`outputs/tiny_imagenet_xla4_summary/`:

```bash
bash scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh
```

On a TPU VM, pass the environment gate through the suite so it stops before
launching any training command if the `.venvxla` stack or TPU visibility is
wrong:

```bash
bash scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh --strict-preflight
```

`--strict-preflight` defaults `PJRT_DEVICE=TPU`, requires visible TPU devices,
requires the visible TPU count to match the xla4 launch size, requires the
active virtualenv name to be `.venvxla`, checks storage roots, and uses
`MIN_FREE_GB=10` unless overridden. If you intentionally use another venv name,
set `VENV_NAME_REQUIRED=<name>`.
Set `SUMMARY_DIR=/path/to/summary` to place those artifacts elsewhere. The
default run writes `manifest.json`, `protocol.txt`, `preflight.md`, `commands.sh`,
`default.csv`, `default.json`, `default_latex_table.tex`,
`last10_median.csv`, `last10_median.json`, and
`last10_median_latex_table.tex`.
`manifest.json` records the suite preset, smoke/full mode, generated command
arguments, git commit, git status, and expected artifact list for later table
provenance checks.
Suite summary arguments such as `--method saliencymix` are forwarded to
preflight, generated commands, status, CSV/JSON, and LaTeX artifacts, so a
single-method refresh does not fail because unrelated methods remain incomplete.

For a quick suite smoke test:

```bash
bash scripts/experiment_run/smoke_tiny_imagenet_xla4.sh
```

The smoke script runs the suite with `--smoke --strict-preflight`. You can pass
summary filters such as `--method saliencymix`; smoke defaults are prepended so
later explicit `--train-arg` overrides still win.
By default, smoke mode writes checkpoints under the isolated root
`./checkpoints/tiny_imagenet_xla4_smoke` instead of the formal table checkpoint
root. Override it with `SMOKE_CHECKPOINT_DIR=...`, or pass an explicit
`--train-arg=--checkpoint-dir` value. Smoke metrics still use the config's
formal output path so status/summary can mark them as incomplete candidates;
a later full run archives those incompatible one-epoch metrics before writing
fresh table metrics.

Smoke metrics are reported as `incomplete` because the summary compares the
recorded epoch against the config's full `epochs: 200`. The full suite will
rerun those rows and `latex-table` will keep their Tiny-ImageNet cells as `--`
until the complete metrics are present. Smoke mode also overrides
SaliencyMix/Guided-SR to `--saliency-source gradient`, so preflight reports
saliency caches as `skipped` and the generated commands do not build
precomputed caches.

If you already trained an older xla4 run and still have
`checkpoints/<run_name>/best.pt`, refresh only the best-checkpoint final test
and current run metadata without rerunning 200 epochs:

```bash
python -m allthemix.cli.summarize --preset tiny-imagenet-xla4 --format commands \
  --train-arg=--eval-only | bash -e
```

Generated eval-only commands keep each YAML's `epochs: 200`, load the default
`checkpoints/<run_name>/best.pt` when present, write an epoch-200 `final_test`
row, and skip SaliencyMix cache builds because no training-time saliency mixing
is executed. Guided-SR has no formal table cache to build. If a row has no
`best.pt`, the generated command
falls back to the full training command for that row; check `--format
preflight --train-arg=--eval-only` to see the split before launching. A manually
launched `--eval-only` run still exits instead of silently evaluating last or
randomly initialized weights when `final_test_checkpoint: best` is configured.
Generated commands reject a manually provided `--checkpoint` unless exactly one
method row is selected; use `--method saliencymix` or another single method so
one checkpoint is not accidentally applied to multiple table rows.
An explicit external best checkpoint is honored for that one row, while an
explicit `last.pt` is replaced by the default `best.pt` when available and is
rejected otherwise under `final_test_checkpoint: best`.
Generated eval-only refresh requires compatible `best.pt` metadata;
best checkpoints from smoke or semantic override runs are treated as missing and
the command generator falls back to full training instead of refreshing table
metadata from incompatible weights. Legacy best checkpoints without metadata are
also treated as missing, because the resolved config includes a
`model_impl_version` stamp and cannot prove that metadata-free weights were
trained with the current network semantics. Corrupt or incomplete `.pt.json`
sidecars are treated as incompatible rather than trusted as legacy checkpoints.
Training final-test evaluation is equally strict: with
`final_test_checkpoint: best`, the trainer must restore in-memory best weights
or `checkpoints/<run_name>/best.pt`; otherwise it fails instead of writing a
last-checkpoint test result into a best-checkpoint table run.
When a manually supplied checkpoint contains embedded config metadata, the
trainer validates that metadata against the current resolved protocol before
using it; if a `.pt.json` sidecar is present, that sidecar is checked first.
Weight-only external checkpoints without config metadata remain available for
explicit manual evaluation.

```bash
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_baseline_xla4.sh --device xla --num-cores 4 --num-workers 0
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_mixup_xla4.sh --device xla --num-cores 4 --num-workers 0
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_cutmix_xla4.sh --device xla --num-cores 4 --num-workers 0
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_resizemix_xla4.sh --device xla --num-cores 4 --num-workers 0
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_fmix_xla4.sh --device xla --num-cores 4 --num-workers 0
bash scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh --num-workers 0
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh --device xla --num-cores 4 --num-workers 0
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh --device xla --num-cores 4 --num-workers 0
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_catchupmix_xla4.sh --device xla --num-cores 4 --num-workers 0
```

The older `*_legacy.yaml` configs preserve the single-process AllTheMix batch
size of `128`; override `--batch-size 32` manually if you launch them on 4 TPU
devices.

To mirror the official FMix Tiny-ImageNet script more closely, use the paper
config: `200 epochs`, global batch `128`, `lr=0.1`, `weight_decay=1e-4`,
step milestones `150,180`, no train validation split, and the official Tiny
validation set each epoch. On 4 TPU devices use the XLA config with per-device
batch `32`:

```bash
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_fmix_paper_xla4.sh --device xla --num-cores 4
```

Run the matching MixUp paper-style Tiny-ImageNet config with the same schedule:

```bash
bash scripts/experiment_run/run_tiny_imagenet_preact_resnet18_mixup_paper_xla4.sh --device xla --num-cores 4
```

`saliencymix_xla4.yaml` defaults to `saliency_source: batch` for table runs,
matching the cached-saliency path used by AllTheMix-style SaliencyMix. Generate
the Tiny-ImageNet train saliency cache once before the full SaliencyMix run. The
SaliencyMix cache builder defaults to OpenCV fine-grained saliency, matching the
AllTheMix saliency preprocessor:

```bash
bash scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh \
  --batch-size 128 --num-workers 4 --device auto
```

Guided-SR table runs use online spectral-residual saliency with the same
saliency-ratio mixer used by the AllTheMix GuidedMixup path. Online SR and
gradient smoke saliency are computed from augmented unit-space images, not from
the normalized tensors fed to the classifier; SR maps are min-max normalized per
image before GuidedMixup's saliency-ratio normalization. A separate
spectral-residual cache script remains available only for debugging or explicit
cached GuidedMixup-style ablations:

```bash
bash scripts/experiment_run/build_tiny_imagenet_guided_sr_cache.sh \
  --batch-size 128 --num-workers 4 --device auto
```

The Tiny-ImageNet Guided-SR table config uses GuidedMixup SR-style method
settings (`guidedmixup_alpha: 1.0`, `guidedmixup_prob: 0.5`,
`guidedmixup_condition: greedy`) while keeping the shared xla4 table training
schedule. With greedy Guided-SR pairing,
`cross_device_shuffle` is disabled for this config; the saliency-ratio mask
uses online spectral-residual maps computed from the augmented local batch after
reversing dataset normalization. As
in the AllTheMix GuidedMixup implementation, `guidedmixup_alpha` is retained
for interface compatibility; the Guided-SR saliency-ratio mask itself does not
sample a beta lambda.

Full SaliencyMix table caches require the OpenCV contrib saliency backend; the
TPU setup script installs `opencv-contrib-python-headless` for this. Cache
generation now fails clearly if that backend is unavailable, so a formal table
run cannot silently fall back to gradient saliency. `--allow-gradient-fallback`
exists only for smoke/debug cache experiments, and summary preflight rejects
those caches for the table protocol.
The SaliencyMix table config uses `basic_aug: false` with
`sal_aug_recipe: tiny_openmixup`; image tensors and cached saliency maps receive
the same random resized crop and horizontal flip before batch mixing. For cached
saliency runs, the train image transform delays normalization until after this
paired spatial augmentation, so the path stays
`raw image -> paired spatial aug -> normalize -> batch mix`.

For quick TPU smoke tests without a cache, override the run with
`--saliency-source gradient`. SaliencyMix and Guided-SR then use cheap
gradient saliency maps on the same augmented unit-space images instead of
compiling the online spectral-residual path.

After the runs finish, summarize the Tiny-ImageNet top-1 error column:

```bash
bash scripts/experiment_run/collect_tiny_imagenet_xla4_results.sh
```

The collect script only reads metrics/config/checkpoint sidecars and does not
require visible TPU devices or an active `.venvxla` by name; it still sets
`PYTHONNOUSERSITE=1`, so run it from a Python environment that has the repo's
normal dependencies installed.
This writes `status.md`, default `csv/json/latex-table`, OpenMixup-style
`last10_median` `csv/json/latex-table`, and `collect_manifest.json` under
`outputs/tiny_imagenet_xla4_summary/`. It does not require every row to be
complete by default, so partial runs still produce inspectable artifacts with
missing table cells as `--`. Add `--require-complete` when you want collection
to fail unless every selected row is table-ready; the completeness gate runs
before writing the final table artifacts and removes stale final table files
from earlier collects, while `collect_manifest.json`, `protocol.txt`, and
`status.md` remain available for debugging.
For the xla4 table protocol, `final_test` is run on the best validation
checkpoint. Missing or incomplete runs are emitted as `--`.
The Tiny-ImageNet table configs require the run directory `config.json` written
by the trainer to match the current resolved protocol; stale metrics produced
before a protocol change are reported as `missing_run_metadata` or
`incompatible_run_config` and are not inserted into the table.
The resolved metadata includes `model_impl_version`; this prevents old
PreActResNet checkpoints or metrics from being reused after architecture
semantics change even when the tensor shapes still load.
When a new table run starts and an existing `metrics.csv` is paired with missing
or incompatible run metadata, the trainer archives the old file as
`metrics.stale-*.csv` before writing fresh metrics.
In CSV/markdown output, `tiny_imagenet_top1_error` is the safe table value and
stays `--` unless the run is complete and protocol-clean; `candidate_top1_error`
keeps the raw selected metric for debugging incomplete runs. CSV rows also
include `protocol_id`, config/metrics paths, resume checkpoint, best checkpoint,
`final_test_checkpoint`, `final_test_checkpoint_source`, and saliency
prerequisite path for downstream table scripts. `latex-table`
preserves the existing CIFAR/STL/Cars/CUB values and fills only the
Tiny-ImageNet column from complete local metrics.
JSON output keeps the same safe table value plus fraction-scale numeric fields,
status, metric source, config path, metrics path, prerequisite path, resume
checkpoint path, best checkpoint path, final-test checkpoint policy, and
final-test checkpoint source. It also records `protocol_id`, the protocol
metadata, and `metric_mode` (`auto` for the default view, or
`last10_median` for the OpenMixup-style view) so generated JSON files remain
self-describing when copied elsewhere.
`latex-table` automatically bolds the lowest complete Tiny-ImageNet top-1 error
and ignores incomplete, invalid-cache, or smoke-run candidates for that
comparison.
The default Tiny-ImageNet table gate also requires an epoch-level validation row
at the configured final epoch, so a lone copied `final_test` row cannot silently
enter the table as a complete 200-epoch run. Eval-only refreshes produced by the
trainer are still accepted because they write both `eval` and `final_test` rows.
For an OpenMixup-style Tiny-ImageNet view, pass
`--metric last10_median`; OpenMixup reports the median top-1 accuracy over the
last 10 training epochs for this benchmark. This mode requires at least 10
epoch-level validation rows; a metrics file with only the final epoch is treated
as incomplete for that reporting style.
`--format commands` prints only missing runs by default, so it is safe after a
TPU reboot or a partial experiment batch. If SaliencyMix or Guided-SR uses
cached saliency and the cache is missing, the generated command list includes
the cache build step before that training step. Partial or invalid cache files
are rebuilt with `--overwrite`, which protects against cache smoke tests made
with `--limit`. Cache builds also write a versioned `.npy.json` metadata
sidecar with the saliency method, dataset, recipe, transform profile, image
size, count, and shape. The formal Tiny-ImageNet cache builder refuses to start
unless the train split contains all 100000 examples; `--limit` is only for
debug caches sampled from a complete local dataset. Training keeps cache startup
fast by default and does not rescan the full cache in every XLA worker; run the
suite/preflight cache checks before formal launches, or set
`validate_saliency_cache_on_load: true` in a debug config to force a full
load-time scan.
The summary gate accepts OpenCV fine-grained caches for SaliencyMix, so stale
caches or caches built with the wrong saliency method are rebuilt after
cache-builder semantic changes. If a
non-complete run has a config-compatible `checkpoints/<run_name>/last.pt`, and
resuming can still produce the required OpenMixup last-10 epoch rows, the
generated training command includes `--checkpoint` so it resumes from the last
completed epoch. Checkpoints write a lightweight `.pt.json` metadata sidecar,
and smoke checkpoints made with overrides such as `--saliency-source gradient`
are not auto-resumed into full table runs. Auto-resume is also disabled for
generated commands that override the training contract, such as smoke commands
with `--epochs`, `--max-train-steps`, `--max-eval-steps`, `--learning-rate`,
`--lr-schedule`, or `--saliency-source`, so quick TPU checks do not
accidentally evaluate an old long-run checkpoint. Path-only overrides
such as `--data-dir` still allow auto-resume, which is useful when the TPU data
mount differs from the config default. Cache build commands inherit safe
dataset/cache context overrides such as `--data-dir`, `--recipe`, and
`--saliency-dir`; they also receive the summary command's
top-level `--num-workers`, so generated cache and training steps use the same
DataLoader worker count. `--saliency-path` is mapped to the cache builder's
`--output` so the generated cache and training run point at the same file. When
the config's saliency cache directory follows `data_dir`, `--data-dir
/path/data` also moves the default cache location there; pass `--saliency-dir`
to keep cache storage separate from the dataset mount. A relative explicit cache
path such as
`--saliency-dir /path/cache --saliency-path maps.npy` is likewise resolved to
`/path/cache/maps.npy` for cache building, preflight, and training. When a
saliency cache path override is present,
generated commands conservatively emit the cache build step for batch-saliency
methods; the builder reuses an already-compatible cache. These
saliency path-only overrides also allow auto-resume; semantic overrides such as
`--seed`, `--epochs`, `--max-eval-steps`, `--learning-rate`, or
`--saliency-source` do not. Path-only cache relocations are also ignored by the
run/checkpoint compatibility check, so changing a TPU mount does not archive
otherwise-compatible metrics. Use `--train-arg=--checkpoint-dir` for temporary
smoke/debug checkpoint roots; because that changes where checkpoint sidecars are
resolved, generated commands disable automatic resume and automatic eval-only
best-checkpoint loading unless you also pass an explicit `--checkpoint`.
`--train-arg` is validated before command emission;
reserved launcher fields such as `--config`, `--device`, `--num-cores`, and
`--num-workers` are rejected there, so use the summary command's top-level
options or the fixed preset configs for those values. `--output-dir` is also
reserved for generated table commands because summaries read the output path
declared in the YAML; edit the config if a persistent output relocation is
needed. `--dataset` is reserved as well because the `tiny-imagenet-xla4` preset
is specifically for Tiny-ImageNet; use `--data-dir` to point at a different
Tiny-ImageNet mount. For this `xla4` preset,
generated XLA commands also require `--num-cores 4`; CPU smoke/debug command
generation is still allowed with other core-count values because the XLA
global-batch check is not active there. To run all missing experiments
sequentially:

```bash
python -m allthemix.cli.summarize --preset tiny-imagenet-xla4 --format commands | bash -e
```

Limit generated summaries or commands to one method with repeatable
`--method`, which is useful when refreshing a manually copied checkpoint:

```bash
python -m allthemix.cli.summarize --preset tiny-imagenet-xla4 --method saliencymix \
  --format commands --train-arg=--eval-only \
  --train-arg=--checkpoint --train-arg=/mnt/checkpoints/saliencymix_best.pt
```

For smoke-command generation:

```bash
python -m allthemix.cli.summarize --preset tiny-imagenet-xla4 --format commands \
  --train-arg=--epochs --train-arg=1 \
  --train-arg=--max-train-steps --train-arg=20 \
  --train-arg=--max-val-steps --train-arg=5 \
  --train-arg=--checkpoint-dir --train-arg=./checkpoints/tiny_imagenet_xla4_smoke \
  --train-arg=--saliency-source --train-arg=gradient
```

Evaluate ImageNet-A with a checkpoint:

```bash
bash scripts/experiment_run/run_imagenet_a_torch_resnet101.sh \
  --checkpoint path/to/imagenet_resnet101_fmix.pt \
  --data-dir data/imagenet-a \
  --device xla \
  --num-cores 4
```

For an ImageNet ResNet-101 MixUp checkpoint, use:

```bash
bash scripts/experiment_run/run_imagenet_a_torch_resnet101_mixup.sh \
  --checkpoint path/to/imagenet_resnet101_mixup.pt \
  --data-dir data/imagenet-a \
  --device xla \
  --num-cores 4
```

ImageNet-A is eval-only and expects ImageFolder wnid class directories. A
1000-class ImageNet head is reduced to the official 200 ImageNet-A classes
before metrics are computed. The `torch_resnet101` config accepts plain
torchvision/official ResNet-101 state dict keys (`conv1`, `layer1`, `fc`, ...)
and maps them into the split backbone/head model.

Tiny-ImageNet accepts either the original `tiny-imagenet-200` layout:

```text
data/tiny-imagenet-200/
  wnids.txt
  train/<wnid>/images/*.JPEG
  val/images/*.JPEG
  val/val_annotations.txt
```

or an ImageFolder-style `train/` and `val/` split:

```text
data/tiny-imagenet-200/
  train/<wnid>/*.JPEG
  val/<wnid>/*.JPEG
```

The class-folder layout may still include `wnids.txt`; the loader and preflight
will use the foldered images when `train/<wnid>/` and `val/<wnid>/` contain
image files.

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
- OpenMixup Tiny-ImageNet configs: https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/tiny_imagenet
