import unittest
import os
import re
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from allthemix.cli.summarize import (
    TINY_IMAGENET_XLA4_PROTOCOL_ID,
    TINY_IMAGENET_XLA4_SPECS,
    expected_resume_config,
    render_commands,
    script_path_for_spec,
    summarize_experiments,
)
from allthemix.cli.train import load_config


class ExperimentScriptTests(unittest.TestCase):
    def _bash_executable(self):
        bash = shutil.which("bash")
        if bash:
            return bash
        for candidate in (
            Path("C:/Program Files/Git/bin/bash.exe"),
            Path("C:/Program Files/Git/usr/bin/bash.exe"),
        ):
            if candidate.exists():
                return str(candidate)
        return None

    def test_shell_scripts_are_bash_syntax_valid_when_bash_is_available(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        scripts = sorted(Path("scripts").rglob("*.sh"))
        self.assertTrue(scripts)
        for path in scripts:
            with self.subTest(path=path.as_posix()):
                subprocess.run([bash, "-n", str(path)], check=True)

    def test_shell_script_config_references_exist(self):
        scripts = sorted(Path("scripts").rglob("*.sh"))
        self.assertTrue(scripts)

        for path in scripts:
            text = path.read_text()
            for match in re.finditer(r"--config\s+([^\s\"\\]+)", text):
                config_path = Path(match.group(1))
                with self.subTest(script=path.as_posix(), config=config_path.as_posix()):
                    self.assertTrue(config_path.exists(), f"missing config referenced by {path}: {config_path}")

    def test_readme_script_references_exist(self):
        readme = Path("README.md").read_text()
        refs = sorted(set(re.findall(r"scripts/[A-Za-z0-9_./-]+\.sh", readme)))
        self.assertTrue(refs)

        for ref in refs:
            with self.subTest(script=ref):
                self.assertTrue(Path(ref).exists(), f"README references missing script: {ref}")

    def test_tiny_imagenet_xla4_runner_scripts_match_table_specs(self):
        for spec in TINY_IMAGENET_XLA4_SPECS:
            with self.subTest(method=spec.method_key):
                path = script_path_for_spec(spec)
                self.assertTrue(path.exists(), f"missing runner script: {path}")
                script = path.read_text()
                self.assertIn("#!/usr/bin/env bash", script)
                self.assertIn("set -euo pipefail", script)
                self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', script)
                self.assertIn('export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"', script)
                self.assertIn("python -m allthemix.cli.train", script)
                self.assertIn(f"--config {spec.config_path.as_posix()}", script)
                self.assertIn("--device xla", script)
                self.assertIn('--num-cores "${NUM_CORES:-4}"', script)
                self.assertIn('--num-workers "${NUM_WORKERS:-0}"', script)
                self.assertIn('"$@"', script)

    def test_all_tiny_imagenet_xla4_train_entrypoints_default_to_xla4_before_user_args(self):
        for path in Path("scripts/experiment_run").glob("run_tiny_imagenet_preact_resnet18_*xla4.sh"):
            with self.subTest(path=path.as_posix()):
                script = path.read_text()
                self.assertIn('export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"', script)
                self.assertRegex(
                    script,
                    r'--device xla \\\n\s+--num-cores "\$\{NUM_CORES:-4\}" \\\n\s+--num-workers "\$\{NUM_WORKERS:-0\}" \\\n\s+"\$@"',
                )

    def test_all_tiny_imagenet_xla4_entrypoints_source_tpu_python_guard(self):
        collect_script = Path("scripts/experiment_run/collect_tiny_imagenet_xla4_results.sh")
        paths = sorted(
            {
                *(path for path in Path("scripts/experiment_run").glob("*xla4*.sh") if path != collect_script),
                Path("scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh"),
                Path("scripts/experiment_run/build_tiny_imagenet_guided_sr_cache.sh"),
            }
        )

        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.as_posix()):
                script = path.read_text()
                self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', script)

    def test_tiny_imagenet_cache_scripts_match_saliency_configs(self):
        saliencymix_script = Path("scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh").read_text()
        guided_sr_script = Path("scripts/experiment_run/build_tiny_imagenet_guided_sr_cache.sh").read_text()

        self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', saliencymix_script)
        self.assertIn("python -m allthemix.cli.build_saliency_cache", saliencymix_script)
        self.assertIn("--config configs/tiny_imagenet/preact_resnet18/saliencymix_xla4.yaml", saliencymix_script)
        self.assertNotIn("--allow-gradient-fallback", saliencymix_script)
        self.assertNotIn("--method gradient", saliencymix_script)
        self.assertIn('"$@"', saliencymix_script)

        self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', guided_sr_script)
        self.assertIn("python -m allthemix.cli.build_saliency_cache", guided_sr_script)
        self.assertIn("--config configs/tiny_imagenet/preact_resnet18/guided_sr_xla4.yaml", guided_sr_script)
        self.assertIn("--method spectral_residual", guided_sr_script)
        self.assertNotIn("--allow-gradient-fallback", guided_sr_script)
        self.assertIn('GUIDED_SR_SALIENCY_OUTPUT:-./data/tiny_imagenet_train_guided_sr_saliency.npy', guided_sr_script)
        self.assertIn('"$@"', guided_sr_script)

    def test_tpu_python_env_guard_disables_user_site_packages(self):
        script = Path("scripts/lib/tpu_python_env.sh").read_text()

        self.assertIn("export PYTHONNOUSERSITE=1", script)
        self.assertIn("stale torch/torch_xla wheels from ~/.local", script)
        self.assertIn('required_venv_name="${VENV_NAME_REQUIRED-.venvxla}"', script)
        self.assertIn("AllTheMiXLA TPU scripts require an active", script)
        self.assertIn("active_venv_name", script)
        self.assertIn("python_prefix", script)
        self.assertIn("sys.prefix", script)

    def test_tpu_python_env_guard_requires_named_virtualenv_when_executed(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        missing = subprocess.run(
            [bash, "-c", "unset VIRTUAL_ENV; source scripts/lib/tpu_python_env.sh"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        matching_script = "\n".join(
            [
                'test_root="${TMPDIR:-/tmp}/allthemixla_env_guard_$$"',
                'trap \'rm -rf "$test_root"\' EXIT',
                'venv="$test_root/.venvxla"',
                'mkdir -p "$venv/bin"',
                'cat > "$venv/bin/python" <<\'PYSH\'',
                "#!/usr/bin/env bash",
                "cat >/dev/null",
                'printf "%s\\n" "$VIRTUAL_ENV"',
                "PYSH",
                'chmod +x "$venv/bin/python"',
                'export VIRTUAL_ENV="$venv"',
                'export PATH="$venv/bin:$PATH"',
                "source scripts/lib/tpu_python_env.sh",
                "echo ok",
            ]
        )
        matching = subprocess.run(
            [bash, "-c", matching_script],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy(),
        )

        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("source .venvxla/bin/activate", missing.stderr)
        self.assertEqual(matching.returncode, 0)
        self.assertIn("ok", matching.stdout)

    def test_tpu_python_env_guard_rejects_wrong_python_prefix(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        mismatch_script = "\n".join(
            [
                'test_root="${TMPDIR:-/tmp}/allthemixla_env_guard_mismatch_$$"',
                'trap \'rm -rf "$test_root"\' EXIT',
                'venv="$test_root/.venvxla"',
                'wrong_prefix="$test_root/wrong-python-prefix"',
                'mkdir -p "$venv/bin" "$wrong_prefix"',
                'cat > "$venv/bin/python" <<\'PYSH\'',
                "#!/usr/bin/env bash",
                "cat >/dev/null",
                'printf "%s\\n" "$WRONG_PYTHON_PREFIX"',
                "PYSH",
                'chmod +x "$venv/bin/python"',
                'export VIRTUAL_ENV="$venv"',
                'export WRONG_PYTHON_PREFIX="$wrong_prefix"',
                'export PATH="$venv/bin:$PATH"',
                "source scripts/lib/tpu_python_env.sh",
            ]
        )
        result = subprocess.run(
            [bash, "-c", mismatch_script],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("require python from .venvxla", result.stderr)
        self.assertIn("python resolves to prefix", result.stderr)

    def test_tpu_python_env_guard_reports_python_prefix_probe_failure(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        failing_python_script = "\n".join(
            [
                'test_root="${TMPDIR:-/tmp}/allthemixla_env_guard_python_fail_$$"',
                'trap \'rm -rf "$test_root"\' EXIT',
                'venv="$test_root/.venvxla"',
                'mkdir -p "$venv/bin"',
                'cat > "$venv/bin/python" <<\'PYSH\'',
                "#!/usr/bin/env bash",
                "cat >/dev/null",
                "exit 7",
                "PYSH",
                'chmod +x "$venv/bin/python"',
                'export VIRTUAL_ENV="$venv"',
                'export PATH="$venv/bin:$PATH"',
                "source scripts/lib/tpu_python_env.sh",
            ]
        )
        result = subprocess.run(
            [bash, "-c", failing_python_script],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
            env=os.environ.copy(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not inspect the active python prefix", result.stderr)

    def test_tpu_python_env_guard_forces_user_site_packages_off(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        script = "\n".join(
            [
                'test_root="${TMPDIR:-/tmp}/allthemixla_env_guard_usersite_$$"',
                'trap \'rm -rf "$test_root"\' EXIT',
                'venv="$test_root/.venvxla"',
                'mkdir -p "$venv/bin"',
                'cat > "$venv/bin/python" <<\'PYSH\'',
                "#!/usr/bin/env bash",
                "cat >/dev/null",
                'printf "%s\\n" "$VIRTUAL_ENV"',
                "PYSH",
                'chmod +x "$venv/bin/python"',
                'export VIRTUAL_ENV="$venv"',
                'export PATH="$venv/bin:$PATH"',
                "source scripts/lib/tpu_python_env.sh",
                'echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"',
            ]
        )
        result = subprocess.run(
            [bash, "-c", script],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "PYTHONNOUSERSITE": "0",
            },
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("PYTHONNOUSERSITE=1", result.stdout)

    def test_gitignore_keeps_tpu_artifacts_out_but_tracks_script_helpers(self):
        gitignore = Path(".gitignore").read_text()

        self.assertIn(".venvxla/", gitignore)
        self.assertIn("/outputs/", gitignore)
        self.assertIn("!scripts/lib/", gitignore)
        self.assertIn("!scripts/lib/*.sh", gitignore)

        tracked_helpers = [
            "scripts/lib/tpu_python_env.sh",
            "scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh",
            "configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml",
        ]
        ignored_artifacts = [
            ".venvxla/bin/python",
            "outputs/tiny_imagenet_preact_resnet18_baseline_xla4/metrics.csv",
            "checkpoints/tiny_imagenet_preact_resnet18_baseline_xla4/best.pt",
            "data/tiny-imagenet-200/wnids.txt",
        ]
        for path in tracked_helpers:
            with self.subTest(tracked=path):
                result = subprocess.run(
                    ["git", "check-ignore", "-q", path],
                    cwd=Path.cwd(),
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0, f"{path} should not be ignored")
        for path in ignored_artifacts:
            with self.subTest(ignored=path):
                result = subprocess.run(
                    ["git", "check-ignore", "-q", path],
                    cwd=Path.cwd(),
                    check=False,
                )
                self.assertEqual(result.returncode, 0, f"{path} should be ignored")

    def test_tpu_setup_script_runs_xla_env_verifier(self):
        script = Path("scripts/setup_tpu_venvxla.sh").read_text()

        self.assertIn("export PYTHONNOUSERSITE=1", script)
        self.assertIn('VENV_NAME="$(basename "$VENV_DIR")"', script)
        self.assertIn('EXPECTED_TPU_DEVICES="${EXPECTED_TPU_DEVICES:-4}"', script)
        self.assertIn('python -m allthemix.cli.verify_xla_env --skip-device-check --require-venv-name "$VENV_NAME"', script)
        self.assertIn(
            'PJRT_DEVICE=TPU python -m allthemix.cli.verify_xla_env --require-tpu --expected-tpu-devices "$EXPECTED_TPU_DEVICES" --require-venv-name "$VENV_NAME"',
            script,
        )
        self.assertIn("After activating the environment, run a smoke check with:", script)

    def test_tpu_setup_script_guards_recreate_delete(self):
        script = Path("scripts/setup_tpu_venvxla.sh").read_text()

        self.assertIn('VENV_REAL="$(cd "$VENV_DIR" && pwd -P)"', script)
        self.assertIn('REPO_REAL="$(pwd -P)"', script)
        self.assertIn("Refusing to recreate VENV_DIR outside this repo", script)
        self.assertIn('rm -rf "$VENV_REAL"', script)

    def test_tpu_setup_script_installs_runtime_python_helpers(self):
        script = Path("scripts/setup_tpu_venvxla.sh").read_text()

        self.assertIn("python -m pip install --no-cache-dir numpy opencv-contrib-python-headless pyyaml pillow", script)
        self.assertNotIn("tqdm", script)

    def test_tpu_setup_script_keeps_pip_temp_and_cache_off_root_disk(self):
        script = Path("scripts/setup_tpu_venvxla.sh").read_text()

        self.assertIn('PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"', script)
        self.assertIn('PIP_TMPDIR="${PIP_TMPDIR:-/dev/shm/allthemixla-pip-tmp}"', script)
        self.assertIn('mkdir -p "$PIP_TMPDIR"', script)
        self.assertIn('export TMPDIR="$PIP_TMPDIR"', script)
        self.assertIn("python -m pip cache purge", script)
        self.assertIn("python -m pip install --no-cache-dir", script)

    def test_tpu_setup_script_installs_data_download_system_tools(self):
        script = Path("scripts/setup_tpu_venvxla.sh").read_text()

        self.assertIn("ca-certificates", script)
        self.assertIn("curl", script)
        self.assertIn("unzip", script)

    def test_shell_scripts_run_from_repo_root(self):
        setup_script = Path("scripts/setup_tpu_venvxla.sh").read_text()
        self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', setup_script)
        self.assertIn('REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"', setup_script)
        self.assertIn('cd "$REPO_ROOT"', setup_script)

        download_script = Path("scripts/download_tiny_imagenet.sh").read_text()
        self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', download_script)
        self.assertIn('REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"', download_script)
        self.assertIn('cd "$REPO_ROOT"', download_script)

        for path in Path("scripts/experiment_run").glob("*.sh"):
            with self.subTest(path=path.as_posix()):
                script = path.read_text()
                self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', script)
                self.assertIn('REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"', script)
                self.assertIn('cd "$REPO_ROOT"', script)

    def test_shell_scripts_keep_lf_line_endings_for_tpu(self):
        scripts = [
            Path("scripts/setup_tpu_venvxla.sh"),
            Path("scripts/download_tiny_imagenet.sh"),
            *Path("scripts/lib").glob("*.sh"),
            *Path("scripts/experiment_run").glob("*.sh"),
        ]

        for path in scripts:
            with self.subTest(path=path.as_posix()):
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_gitattributes_forces_lf_for_shell_scripts(self):
        attributes = Path(".gitattributes").read_text()

        self.assertIn("*.sh text eol=lf", attributes)

    def test_requirements_include_direct_runtime_dependencies(self):
        requirements = {
            line.strip().lower()
            for line in Path("requirements.txt").read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        self.assertIn("numpy", requirements)
        self.assertIn("opencv-contrib-python-headless", requirements)
        self.assertIn("pillow", requirements)
        self.assertIn("pyyaml", requirements)
        self.assertIn("torch", requirements)
        self.assertIn("torchvision", requirements)

    def test_tiny_imagenet_download_script_prepares_default_data_layout(self):
        script = Path("scripts/download_tiny_imagenet.sh").read_text()

        self.assertIn("tiny-imagenet-200.zip", script)
        self.assertIn("https://cs231n.stanford.edu/tiny-imagenet-200.zip", script)
        self.assertIn("90528d7ca1a48142e341f4ef8d21d0de", script)
        self.assertIn('data_dir="${DATA_DIR:-./data}"', script)
        self.assertIn('tiny_root="$data_dir/tiny-imagenet-200"', script)
        self.assertIn("--md5 MD5", script)
        self.assertIn("--skip-md5", script)
        self.assertIn("skip_md5=false", script)
        self.assertIn('expected_train_images="${TINY_EXPECTED_TRAIN_IMAGES:-100000}"', script)
        self.assertIn('expected_val_images="${TINY_EXPECTED_VAL_IMAGES:-10000}"', script)
        self.assertIn("count_images()", script)
        self.assertIn("tiny_imagenet_ready()", script)
        self.assertIn('temp_archive_path="$data_dir/.$archive_name.tmp"', script)
        self.assertIn('rm -f "$temp_archive_path"', script)
        self.assertIn('curl -fL "$tiny_url" -o "$temp_archive_path"', script)
        self.assertIn('wget -O "$temp_archive_path" "$tiny_url"', script)
        self.assertIn('mv "$temp_archive_path" "$archive_path"', script)
        self.assertIn('if [[ "$skip_md5" != true && -n "$expected_md5" ]]', script)
        self.assertIn('unzip -q -o "$archive_path" -d "$data_dir"', script)
        self.assertIn("Expected image counts: train=$expected_train_images val=$expected_val_images", script)

    def test_tiny_imagenet_download_script_does_not_reuse_partial_layout(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        unzip_check = subprocess.run(
            [bash, "-lc", "command -v unzip >/dev/null 2>&1"],
            text=True,
            capture_output=True,
            check=False,
        )
        if unzip_check.returncode != 0:
            self.skipTest("unzip is not available under bash")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            tiny_root = data_dir / "tiny-imagenet-200"
            (tiny_root / "train").mkdir(parents=True)
            (tiny_root / "val").mkdir(parents=True)
            with zipfile.ZipFile(data_dir / "tiny-imagenet-200.zip", "w") as archive:
                archive.writestr("tiny-imagenet-200/train/n00000001/images/train_0.JPEG", "train")
                archive.writestr("tiny-imagenet-200/val/images/val_0.JPEG", "val")

            result = subprocess.run(
                [
                    bash,
                    "scripts/download_tiny_imagenet.sh",
                    "--data-dir",
                    str(data_dir),
                    "--skip-md5",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "TINY_EXPECTED_TRAIN_IMAGES": "1",
                    "TINY_EXPECTED_VAL_IMAGES": "1",
                },
            )

            extracted_train_exists = (tiny_root / "train" / "n00000001" / "images" / "train_0.JPEG").exists()
            extracted_val_exists = (tiny_root / "val" / "images" / "val_0.JPEG").exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(extracted_train_exists)
        self.assertTrue(extracted_val_exists)
        self.assertIn("Tiny-ImageNet is ready", result.stdout)

    def test_tiny_imagenet_xla4_preflight_script_runs_strict_tpu_gate(self):
        script = Path("scripts/experiment_run/preflight_tiny_imagenet_xla4.sh").read_text()

        self.assertIn("python -m allthemix.cli.summarize", script)
        self.assertIn("--preset tiny-imagenet-xla4", script)
        self.assertIn("--format preflight", script)
        self.assertIn("--check-env", script)
        self.assertIn("--require-tpu-env", script)
        self.assertIn('required_venv_name="${VENV_NAME_REQUIRED-.venvxla}"', script)
        self.assertIn('if [[ -n "$required_venv_name" ]]; then', script)
        self.assertIn('preflight_args+=(--require-venv-name "$required_venv_name")', script)
        self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', script)
        self.assertIn('export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"', script)
        self.assertIn('--min-free-gb "${MIN_FREE_GB:-10}"', script)
        self.assertIn("--require-existing-storage-roots", script)
        self.assertIn("--require-complete", script)
        self.assertIn('"$@"', script)

    def test_tiny_imagenet_xla4_prepare_script_downloads_then_strict_preflights(self):
        script = Path("scripts/experiment_run/prepare_tiny_imagenet_xla4.sh").read_text()

        self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', script)
        self.assertIn('"$bash_cmd" scripts/download_tiny_imagenet.sh', script)
        self.assertIn('"$bash_cmd" scripts/experiment_run/preflight_tiny_imagenet_xla4.sh', script)
        self.assertIn('data_dir="${DATA_DIR:-./data}"', script)
        self.assertIn('bash_cmd="${BASH_CMD:-bash}"', script)
        self.assertIn('download_args+=("--data-dir" "$2")', script)
        self.assertIn('preflight_args+=("--train-arg=--data-dir" "--train-arg=$2")', script)
        self.assertIn('download_args+=("--data-dir" "$data_dir")', script)
        self.assertIn('preflight_args+=("--train-arg=--data-dir" "--train-arg=$data_dir")', script)
        self.assertIn("Everything after -- is forwarded", script)
        self.assertIn("--skip-md5", script)
        self.assertIn("--force", script)

    def test_prepare_script_keeps_user_preflight_args_after_default_data_dir(self):
        script = Path("scripts/experiment_run/prepare_tiny_imagenet_xla4.sh").read_text()
        after_separator = script.split("    --)\n", 1)[1].split("      ;;\n", 1)[0]

        self.assertLess(
            after_separator.index('preflight_args+=("--train-arg=--data-dir" "--train-arg=$data_dir")'),
            after_separator.index('preflight_args+=("$@")'),
        )
        self.assertIn("data_dir_forwarded=true", after_separator)

    def test_prepare_script_executes_download_then_preflight_with_split_args(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        def msys_path(path: Path) -> str:
            value = path.resolve().as_posix()
            if len(value) >= 3 and value[1:3] == ":/":
                return f"/{value[0].lower()}{value[2:]}"
            return value

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            call_log = root / "bash_calls.txt"
            fake_bash = fake_bin / "bash"
            fake_bash.write_text(
                "\n".join(
                    [
                        "#!/bin/sh",
                        'printf "%s" "$1" >> "$CALL_LOG"',
                        "shift",
                        'for arg in "$@"; do printf " <%s>" "$arg" >> "$CALL_LOG"; done',
                        'printf "\\n" >> "$CALL_LOG"',
                    ]
                )
                + "\n"
            )
            os.chmod(fake_bash, 0o755)

            result = subprocess.run(
                [
                    bash,
                    "scripts/experiment_run/prepare_tiny_imagenet_xla4.sh",
                    "--data-dir",
                    "/mnt/tiny",
                    "--skip-md5",
                    "--",
                    "--method",
                    "fmix",
                    "--train-arg=--saliency-dir",
                    "--train-arg=/mnt/cache",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "CALL_LOG": msys_path(call_log),
                    "BASH_CMD": msys_path(fake_bash),
                    "VIRTUAL_ENV": "/tmp/AllTheMiXLA/.venvxla",
                    "VENV_NAME_REQUIRED": "",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = call_log.read_text().splitlines()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], "scripts/download_tiny_imagenet.sh <--data-dir> </mnt/tiny> <--skip-md5>")
        self.assertEqual(
            calls[1],
            "scripts/experiment_run/preflight_tiny_imagenet_xla4.sh "
            "<--train-arg=--data-dir> <--train-arg=/mnt/tiny> "
            "<--method> <fmix> <--train-arg=--saliency-dir> <--train-arg=/mnt/cache>",
        )

    def test_tiny_imagenet_xla4_smoke_script_runs_strict_smoke_suite(self):
        script = Path("scripts/experiment_run/smoke_tiny_imagenet_xla4.sh").read_text()

        self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', script)
        self.assertIn('bash_cmd="${BASH_CMD:-bash}"', script)
        self.assertIn('"$bash_cmd" scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh', script)
        self.assertIn("--smoke", script)
        self.assertIn("--strict-preflight", script)
        self.assertIn('"$@"', script)

    def test_tiny_imagenet_xla4_smoke_script_executes_suite_with_forwarded_args(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        def msys_path(path: Path) -> str:
            value = path.resolve().as_posix()
            if len(value) >= 3 and value[1:3] == ":/":
                return f"/{value[0].lower()}{value[2:]}"
            return value

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            call_log = root / "bash_calls.txt"
            fake_bash = fake_bin / "bash"
            fake_bash.write_text(
                "\n".join(
                    [
                        "#!/bin/sh",
                        'printf "%s" "$1" >> "$CALL_LOG"',
                        "shift",
                        'for arg in "$@"; do printf " <%s>" "$arg" >> "$CALL_LOG"; done',
                        'printf "\\n" >> "$CALL_LOG"',
                    ]
                )
                + "\n"
            )
            os.chmod(fake_bash, 0o755)

            result = subprocess.run(
                [
                    bash,
                    "scripts/experiment_run/smoke_tiny_imagenet_xla4.sh",
                    "--method",
                    "fmix",
                    "--train-arg=--log-interval",
                    "--train-arg=0",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "CALL_LOG": msys_path(call_log),
                    "BASH_CMD": msys_path(fake_bash),
                    "VIRTUAL_ENV": "/tmp/AllTheMiXLA/.venvxla",
                    "VENV_NAME_REQUIRED": "",
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = call_log.read_text().splitlines()

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0],
            "scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh "
            "<--smoke> <--strict-preflight> <--method> <fmix> "
            "<--train-arg=--log-interval> <--train-arg=0>",
        )

    def test_tiny_imagenet_xla4_collect_script_writes_table_artifacts(self):
        script = Path("scripts/experiment_run/collect_tiny_imagenet_xla4_results.sh").read_text()

        self.assertIn("collect_manifest.json", script)
        self.assertNotIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', script)
        self.assertIn("export PYTHONNOUSERSITE=1", script)
        self.assertIn("does not touch TPU devices", script)
        self.assertIn('"mode": "collect"', script)
        self.assertIn('require_complete=false', script)
        self.assertIn('collect_args=()', script)
        self.assertIn('--require-complete)', script)
        self.assertIn('"require_complete": require_complete', script)
        self.assertIn('"${collect_args[@]}" | tee "$summary_dir/protocol.txt"', script)
        self.assertIn("--format protocol", script)
        self.assertIn("--format status", script)
        self.assertIn("--format csv", script)
        self.assertIn("--format json", script)
        self.assertIn("--format latex-table", script)
        self.assertIn("--metric last10_median", script)
        self.assertIn('tee "$summary_dir/default.csv"', script)
        self.assertIn('tee "$summary_dir/default.json"', script)
        self.assertIn('tee "$summary_dir/default_latex_table.tex"', script)
        self.assertIn('tee "$summary_dir/last10_median.csv"', script)
        self.assertIn('tee "$summary_dir/last10_median.json"', script)
        self.assertIn('tee "$summary_dir/last10_median_latex_table.tex"', script)
        self.assertIn('if [[ "$require_complete" == true ]]; then', script)
        self.assertIn("final_artifacts=(", script)
        self.assertIn('rm -f "$summary_dir/$artifact"', script)
        self.assertIn("--require-complete", script)
        self.assertIn("--metric last10_median", script)
        self.assertIn('"${collect_args[@]}" >/dev/null', script)
        self.assertLess(
            script.index('if [[ "$require_complete" == true ]]; then'),
            script.index('tee "$summary_dir/default.csv"'),
        )
        self.assertNotIn('"$@"', script)

    def test_tiny_imagenet_xla4_collect_executes_require_complete_arg_forwarding(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            call_log = root / "python_calls.txt"
            fake_python = fake_bin / "python"
            fake_python.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'printf "%q " "$@" >> "$CALL_LOG"',
                        'printf "\\n" >> "$CALL_LOG"',
                        'if [[ "${1:-}" == "-" ]]; then',
                        "  shift",
                        '  "$REAL_PYTHON" - "$@"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format protocol "* ]]; then',
                        '  echo "tiny-imagenet-xla4 protocol: ok"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format status "* ]]; then',
                        '  echo "| Method | Status |"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format json "* ]]; then',
                        '  echo "{\"rows\": []}"',
                        "  exit 0",
                        "fi",
                        'echo "protocol_id,type,method"',
                    ]
                )
                + "\n"
            )
            os.chmod(fake_python, 0o755)

            result = subprocess.run(
                [
                    bash,
                    "scripts/experiment_run/collect_tiny_imagenet_xla4_results.sh",
                    "--require-complete",
                    "--method",
                    "saliencymix",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                    "CALL_LOG": str(call_log),
                    "SUMMARY_DIR": str(root / "summary"),
                    "VIRTUAL_ENV": "/tmp/AllTheMiXLA/.venvxla",
                    "VENV_NAME_REQUIRED": "",
                    "REAL_PYTHON": sys.executable,
                },
            )

            calls = call_log.read_text().splitlines()
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "summary" / "collect_manifest.json").read_text())
            default_csv_exists = (root / "summary" / "default.csv").exists()
            last10_latex_exists = (root / "summary" / "last10_median_latex_table.tex").exists()

        self.assertTrue(default_csv_exists)
        self.assertTrue(last10_latex_exists)
        self.assertTrue(manifest["require_complete"])
        self.assertEqual(manifest["extra_command_args"], ["--method", "saliencymix"])
        summarize_calls = [line for line in calls if "-m allthemix.cli.summarize" in line]
        self.assertEqual(len(summarize_calls), 10)
        for call in summarize_calls:
            with self.subTest(call=call):
                self.assertIn("--method saliencymix", call)
        require_complete_calls = [line for line in summarize_calls if "--require-complete" in line]
        self.assertEqual(len(require_complete_calls), 2)
        self.assertTrue(all("--format csv" in line for line in require_complete_calls))

    def test_tiny_imagenet_xla4_collect_removes_stale_tables_when_require_complete_fails(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            summary_dir = root / "summary"
            summary_dir.mkdir()
            stale_paths = [
                summary_dir / "default.csv",
                summary_dir / "default.json",
                summary_dir / "default_latex_table.tex",
                summary_dir / "last10_median.csv",
                summary_dir / "last10_median.json",
                summary_dir / "last10_median_latex_table.tex",
            ]
            for path in stale_paths:
                path.write_text("stale\n")

            fake_python = fake_bin / "python"
            fake_python.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'if [[ "${1:-}" == "-" ]]; then',
                        "  shift",
                        '  "$REAL_PYTHON" - "$@"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format protocol "* ]]; then',
                        '  echo "tiny-imagenet-xla4 protocol: ok"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format status "* ]]; then',
                        '  echo "| Method | Status |"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --require-complete "* ]]; then',
                        '  echo "incomplete" >&2',
                        "  exit 9",
                        "fi",
                        'echo "protocol_id,type,method"',
                    ]
                )
                + "\n"
            )
            os.chmod(fake_python, 0o755)

            result = subprocess.run(
                [
                    bash,
                    "scripts/experiment_run/collect_tiny_imagenet_xla4_results.sh",
                    "--require-complete",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                    "SUMMARY_DIR": str(summary_dir),
                    "REAL_PYTHON": sys.executable,
                },
            )

            manifest_exists = (summary_dir / "collect_manifest.json").exists()
            protocol_exists = (summary_dir / "protocol.txt").exists()
            status_exists = (summary_dir / "status.md").exists()
            stale_exists = [path.exists() for path in stale_paths]

        self.assertEqual(result.returncode, 9)
        self.assertTrue(manifest_exists)
        self.assertTrue(protocol_exists)
        self.assertTrue(status_exists)
        self.assertFalse(any(stale_exists))

    def test_tiny_imagenet_xla4_suite_runs_protocol_commands_and_summary(self):
        script = Path("scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh").read_text()

        self.assertIn('source "$REPO_ROOT/scripts/lib/tpu_python_env.sh"', script)
        self.assertIn("--format protocol", script)
        self.assertIn("--format preflight", script)
        self.assertIn("--format commands", script)
        self.assertIn("--format csv", script)
        self.assertIn("--format json", script)
        self.assertIn("--format latex-table", script)
        self.assertIn("--metric last10_median", script)
        self.assertIn("--require-complete", script)
        self.assertIn("--smoke", script)
        self.assertIn("--strict-preflight", script)
        self.assertIn('while [[ "$#" -gt 0 ]]; do', script)
        self.assertIn('case "$1" in', script)
        self.assertIn('user_extra_args=("${extra_args[@]}")', script)
        self.assertIn(
            'smoke_checkpoint_dir="${SMOKE_CHECKPOINT_DIR:-./checkpoints/tiny_imagenet_xla4_smoke}"',
            script,
        )
        self.assertIn('extra_args+=("${user_extra_args[@]}")', script)
        self.assertIn('export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"', script)
        self.assertLess(
            script.index('export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"'),
            script.index('if [[ "$strict_preflight" == true ]]'),
        )
        self.assertIn("--check-env", script)
        self.assertIn("--require-tpu-env", script)
        self.assertIn('required_venv_name="${VENV_NAME_REQUIRED-.venvxla}"', script)
        self.assertIn('if [[ -n "$required_venv_name" ]]; then', script)
        self.assertIn('preflight_extra_args+=(--require-venv-name "$required_venv_name")', script)
        self.assertIn('--min-free-gb "${MIN_FREE_GB:-10}"', script)
        self.assertIn("--require-existing-storage-roots", script)
        self.assertIn('preflight_extra_args=("${extra_args[@]}")', script)
        self.assertIn('preflight_extra_args=(\n    --check-env', script)
        self.assertIn('preflight_extra_args+=("${extra_args[@]}")', script)
        self.assertIn('"strict_preflight": strict_preflight', script)
        self.assertNotIn('if [[ "${1:-}" == "--smoke" ]]', script)
        self.assertIn("--train-arg=--saliency-source --train-arg=gradient", script)
        self.assertNotIn("--skip-opencv-check", script)
        self.assertIn('summary_dir="${SUMMARY_DIR:-outputs/tiny_imagenet_xla4_summary}"', script)
        self.assertIn('"$summary_dir/manifest.json"', script)
        self.assertIn('"preset": "tiny-imagenet-xla4"', script)
        self.assertIn("TINY_IMAGENET_XLA4_PROTOCOL_ID", script)
        self.assertIn("TINY_IMAGENET_XLA4_PROTOCOL", script)
        self.assertEqual(TINY_IMAGENET_XLA4_PROTOCOL_ID, "allthemix_split200_openmixup_aug_bestval_xla4")
        self.assertIn("OpenMixup's public Tiny-ImageNet", script)
        self.assertIn('"git_commit": git_output("rev-parse", "HEAD")', script)
        self.assertIn('"git_status_short": (git_output("status", "--short") or "").splitlines()', script)
        self.assertIn('tee "$summary_dir/protocol.txt"', script)
        self.assertIn('tee "$summary_dir/preflight.md"', script)
        self.assertIn('"${preflight_extra_args[@]}" | tee "$summary_dir/preflight.md"', script)
        self.assertIn('tee "$summary_dir/commands.sh" | bash -e', script)
        self.assertIn('tee "$summary_dir/status.md"', script)
        self.assertIn('tee "$summary_dir/default.csv"', script)
        self.assertIn('tee "$summary_dir/default.json"', script)
        self.assertIn('tee "$summary_dir/default_latex_table.tex"', script)
        self.assertIn('tee "$summary_dir/last10_median.csv"', script)
        self.assertIn('tee "$summary_dir/last10_median.json"', script)
        self.assertIn('tee "$summary_dir/last10_median_latex_table.tex"', script)

    def test_tiny_imagenet_xla4_suite_keeps_strict_preflight_args_scoped(self):
        script = Path("scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh").read_text()

        for artifact in [
            "protocol.txt",
            "commands.sh",
            "status.md",
            "default.csv",
            "default.json",
            "default_latex_table.tex",
            "last10_median.csv",
            "last10_median.json",
            "last10_median_latex_table.tex",
        ]:
            with self.subTest(artifact=artifact):
                self.assertIn(f'"${{extra_args[@]}}" | tee "$summary_dir/{artifact}', script)
        self.assertIn('"${preflight_extra_args[@]}" | tee "$summary_dir/preflight.md"', script)

    def test_tiny_imagenet_xla4_suite_executes_smoke_strict_arg_forwarding(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            call_log = root / "python_calls.txt"
            fake_python = fake_bin / "python"
            fake_python.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'printf "%q " "$@" >> "$CALL_LOG"',
                        'printf "\\n" >> "$CALL_LOG"',
                        'if [[ "${1:-}" == "-" ]]; then',
                        "  cat >/dev/null",
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format commands "* ]]; then',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format preflight "* ]]; then',
                        '  echo "| check | status | detail |"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format status "* ]]; then',
                        '  echo "| Method | Status |"',
                        "  exit 0",
                        "fi",
                        'echo "tiny-imagenet-xla4 protocol: ok"',
                    ]
                )
                + "\n"
            )
            os.chmod(fake_python, 0o755)

            result = subprocess.run(
                [
                    bash,
                    "scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh",
                    "--smoke",
                    "--strict-preflight",
                    "--train-arg=--log-interval",
                    "--train-arg=0",
                    "--train-arg=--checkpoint-dir",
                    "--train-arg=manual_smoke_checkpoints",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                    "CALL_LOG": str(call_log),
                    "SUMMARY_DIR": str(root / "summary"),
                    "VIRTUAL_ENV": "/tmp/AllTheMiXLA/.venvxla",
                    "VENV_NAME_REQUIRED": "",
                    "MIN_FREE_GB": "10",
                },
            )

            calls = call_log.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        summarize_calls = [line for line in calls if "-m allthemix.cli.summarize" in line]
        self.assertEqual(len(summarize_calls), 4)
        for call in summarize_calls:
            with self.subTest(call=call):
                self.assertIn("--train-arg=--epochs", call)
                self.assertIn("--train-arg=1", call)
                self.assertIn("--train-arg=--max-train-steps", call)
                self.assertIn("--train-arg=20", call)
                self.assertIn("--train-arg=--max-val-steps", call)
                self.assertIn("--train-arg=5", call)
                self.assertIn("--train-arg=--checkpoint-dir", call)
                self.assertIn("--train-arg=./checkpoints/tiny_imagenet_xla4_smoke", call)
                self.assertIn("--train-arg=manual_smoke_checkpoints", call)
                self.assertLess(
                    call.index("--train-arg=./checkpoints/tiny_imagenet_xla4_smoke"),
                    call.index("--train-arg=manual_smoke_checkpoints"),
                )
                self.assertIn("--train-arg=--saliency-source", call)
                self.assertIn("--train-arg=gradient", call)
                self.assertIn("--train-arg=--log-interval", call)
                self.assertIn("--train-arg=0", call)
                if "--format preflight" in call:
                    self.assertIn("--check-env", call)
                    self.assertIn("--require-tpu-env", call)
                    self.assertNotIn("--require-venv-name", call)
                    self.assertIn("--min-free-gb 10", call)
                    self.assertIn("--require-existing-storage-roots", call)
                else:
                    self.assertNotIn("--check-env", call)
                    self.assertNotIn("--require-tpu-env", call)
                    self.assertNotIn("--require-venv-name", call)
                    self.assertNotIn("--min-free-gb", call)
                    self.assertNotIn("--require-existing-storage-roots", call)

    def test_tiny_imagenet_xla4_suite_forwards_storage_overrides_to_all_full_run_steps(self):
        bash = self._bash_executable()
        if bash is None:
            self.skipTest("bash is not available on this host")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            call_log = root / "python_calls.txt"
            fake_python = fake_bin / "python"
            fake_python.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'printf "%q " "$@" >> "$CALL_LOG"',
                        'printf "\\n" >> "$CALL_LOG"',
                        'if [[ "${1:-}" == "-" ]]; then',
                        "  shift",
                        '  "$REAL_PYTHON" - "$@"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format commands "* ]]; then',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format preflight "* ]]; then',
                        '  echo "| check | status | detail |"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format json "* ]]; then',
                        '  echo "{\"rows\": []}"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format csv "* ]]; then',
                        '  echo "protocol_id,type,method"',
                        "  exit 0",
                        "fi",
                        'if [[ " $* " == *" --format latex-table "* ]]; then',
                        '  echo "\\\\begin{tabular}{lll}"',
                        "  exit 0",
                        "fi",
                        'echo "tiny-imagenet-xla4 protocol: ok"',
                    ]
                )
                + "\n"
            )
            os.chmod(fake_python, 0o755)

            storage_args = [
                "--train-arg=--data-dir",
                "--train-arg=/mnt/tiny",
                "--train-arg=--saliency-dir",
                "--train-arg=/mnt/cache",
            ]
            result = subprocess.run(
                [
                    bash,
                    "scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh",
                    "--strict-preflight",
                    *storage_args,
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                    "CALL_LOG": str(call_log),
                    "SUMMARY_DIR": str(root / "summary"),
                    "VIRTUAL_ENV": "/tmp/AllTheMiXLA/.venvxla",
                    "VENV_NAME_REQUIRED": "",
                    "MIN_FREE_GB": "12",
                    "REAL_PYTHON": sys.executable,
                    "MSYS_NO_PATHCONV": "1",
                },
            )

            calls = call_log.read_text().splitlines()
            manifest = json.loads((root / "summary" / "manifest.json").read_text())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(manifest["smoke"])
        self.assertTrue(manifest["strict_preflight"])
        self.assertEqual(manifest["extra_command_args"], storage_args)

        summarize_calls = [line for line in calls if "-m allthemix.cli.summarize" in line]
        self.assertEqual(len(summarize_calls), 9)
        for call in summarize_calls:
            with self.subTest(call=call):
                self.assertIn("--train-arg=--data-dir", call)
                self.assertIn("--train-arg=/mnt/tiny", call)
                self.assertIn("--train-arg=--saliency-dir", call)
                self.assertIn("--train-arg=/mnt/cache", call)
                if "--format preflight" in call:
                    self.assertIn("--check-env", call)
                    self.assertIn("--require-tpu-env", call)
                    self.assertNotIn("--require-venv-name", call)
                    self.assertIn("--min-free-gb 12", call)
                    self.assertIn("--require-existing-storage-roots", call)
                else:
                    self.assertNotIn("--check-env", call)
                    self.assertNotIn("--require-tpu-env", call)
                    self.assertNotIn("--require-venv-name", call)
                    self.assertNotIn("--min-free-gb", call)
                    self.assertNotIn("--require-existing-storage-roots", call)

    def test_summarize_commands_ignores_suite_preflight_flags_but_keeps_train_args(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "allthemix.cli.summarize",
                "--preset",
                "tiny-imagenet-xla4",
                "--format",
                "commands",
                "--device",
                "xla",
                "--num-cores",
                "4",
                "--num-workers",
                "0",
                "--check-env",
                "--require-tpu-env",
                "--require-venv-name",
                ".venvxla",
                "--min-free-gb",
                "10",
                "--require-existing-storage-roots",
                "--train-arg=--epochs",
                "--train-arg=1",
                "--train-arg=--max-train-steps",
                "--train-arg=20",
                "--train-arg=--max-val-steps",
                "--train-arg=5",
                "--train-arg=--saliency-source",
                "--train-arg=gradient",
            ],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = result.stdout
        self.assertIn("--epochs 1 --max-train-steps 20 --max-val-steps 5 --saliency-source gradient", commands)
        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands)
        command_lines = [line for line in commands.splitlines() if line.strip()]
        baseline_command = next(line for line in command_lines if "run_tiny_imagenet_preact_resnet18_baseline_xla4.sh" in line)
        saliencymix_command = next(line for line in command_lines if "run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh" in line)
        guided_sr_command = next(line for line in command_lines if "run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh" in line)
        self.assertNotIn("--saliency-source gradient", baseline_command)
        self.assertIn("--saliency-source gradient", saliencymix_command)
        self.assertIn("--saliency-source gradient", guided_sr_command)
        self.assertNotIn("build_tiny_imagenet_saliencymix_cache.sh", commands)
        self.assertNotIn("build_tiny_imagenet_guided_sr_cache.sh", commands)
        for summary_only_arg in (
            "--check-env",
            "--require-tpu-env",
            "--require-venv-name",
            "--min-free-gb",
            "--require-existing-storage-roots",
        ):
            with self.subTest(summary_only_arg=summary_only_arg):
                self.assertNotIn(summary_only_arg, commands)

    def test_summarize_cli_formal_commands_build_only_saliencymix_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for spec in TINY_IMAGENET_XLA4_SPECS:
                target = root / spec.config_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(spec.config_path.read_text())

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "allthemix.cli.summarize",
                    "--root",
                    str(root),
                    "--preset",
                    "tiny-imagenet-xla4",
                    "--format",
                    "commands",
                    "--device",
                    "xla",
                    "--num-cores",
                    "4",
                    "--num-workers",
                    "0",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.strip()]
        train_commands = [line for line in commands if "run_tiny_imagenet_preact_resnet18_" in line]
        cache_commands = [line for line in commands if "build_tiny_imagenet_" in line]

        self.assertEqual(len(train_commands), len(TINY_IMAGENET_XLA4_SPECS))
        self.assertEqual(
            cache_commands,
            ["bash scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh --num-workers 0"],
        )
        saliency_cache_index = commands.index(cache_commands[0])
        saliencymix_index = next(
            index
            for index, command in enumerate(commands)
            if "run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh" in command
        )
        guided_sr_index = next(
            index
            for index, command in enumerate(commands)
            if "run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh" in command
        )
        self.assertLess(saliency_cache_index, saliencymix_index)
        self.assertLess(saliencymix_index, guided_sr_index)
        self.assertNotIn("build_tiny_imagenet_guided_sr_cache.sh", result.stdout)
        self.assertNotIn("--saliency-source", result.stdout)

    def test_summarize_cli_formal_commands_keep_data_and_saliency_storage_overrides_aligned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for spec in TINY_IMAGENET_XLA4_SPECS:
                target = root / spec.config_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(spec.config_path.read_text())

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "allthemix.cli.summarize",
                    "--root",
                    str(root),
                    "--preset",
                    "tiny-imagenet-xla4",
                    "--format",
                    "commands",
                    "--device",
                    "xla",
                    "--num-cores",
                    "4",
                    "--num-workers",
                    "0",
                    "--train-arg=--data-dir",
                    "--train-arg=/mnt/tiny",
                    "--train-arg=--saliency-dir",
                    "--train-arg=/mnt/cache",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.strip()]
        cache_commands = [line for line in commands if "build_tiny_imagenet_" in line]
        baseline_command = next(
            line for line in commands if "run_tiny_imagenet_preact_resnet18_baseline_xla4.sh" in line
        )
        saliencymix_command = next(
            line for line in commands if "run_tiny_imagenet_preact_resnet18_saliencymix_xla4.sh" in line
        )
        guided_sr_command = next(
            line for line in commands if "run_tiny_imagenet_preact_resnet18_guided_sr_xla4.sh" in line
        )

        self.assertEqual(
            cache_commands,
            [
                "bash scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh "
                "--num-workers 0 --data-dir /mnt/tiny --saliency-dir /mnt/cache"
            ],
        )
        self.assertIn("--data-dir /mnt/tiny", baseline_command)
        self.assertNotIn("--saliency-dir /mnt/cache", baseline_command)
        self.assertIn("--data-dir /mnt/tiny", saliencymix_command)
        self.assertIn("--saliency-dir /mnt/cache", saliencymix_command)
        self.assertIn("--data-dir /mnt/tiny", guided_sr_command)
        self.assertIn("--saliency-dir /mnt/cache", guided_sr_command)
        self.assertNotIn("build_tiny_imagenet_guided_sr_cache.sh", result.stdout)

    def test_summarize_cli_formal_commands_accept_equals_style_train_arg_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for spec in TINY_IMAGENET_XLA4_SPECS:
                target = root / spec.config_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(spec.config_path.read_text())

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "allthemix.cli.summarize",
                    "--root",
                    str(root),
                    "--preset",
                    "tiny-imagenet-xla4",
                    "--format",
                    "commands",
                    "--device",
                    "xla",
                    "--num-cores",
                    "4",
                    "--num-workers",
                    "0",
                    "--train-arg=--data-dir=/mnt/tiny",
                    "--train-arg=--saliency-dir=/mnt/cache",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-dir=/mnt/tiny", result.stdout)
        self.assertIn("--saliency-dir=/mnt/cache", result.stdout)
        self.assertIn(
            "bash scripts/experiment_run/build_tiny_imagenet_saliencymix_cache.sh "
            "--num-workers 0 --data-dir /mnt/tiny --saliency-dir /mnt/cache",
            result.stdout,
        )
        self.assertNotIn("build_tiny_imagenet_guided_sr_cache.sh", result.stdout)

    def test_summarize_cli_include_complete_eval_only_uses_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = Path("configs/tiny_imagenet/preact_resnet18/baseline_xla4.yaml")
            run_name = "tiny_imagenet_preact_resnet18_baseline_xla4"
            (root / config_path.parent).mkdir(parents=True)
            (root / config_path).write_text(
                "\n".join(
                    [
                        "dataset: tiny_imagenet",
                        "data_dir: ./data",
                        "model: preact_resnet18",
                        "method: baseline",
                        "batch_size: 32",
                        "global_batch_size: 128",
                        "epochs: 200",
                        "learning_rate: 0.1",
                        "momentum: 0.9",
                        "weight_decay: 0.0005",
                        "lr_schedule: step",
                        "lr_decay_epochs: [150, 180]",
                        "basic_aug: false",
                        "aug_recipe: tiny_openmixup",
                        "validation_split: 0.1",
                        "eval_on_test_each_epoch: false",
                        "final_test: true",
                        "final_test_checkpoint: best",
                        "output_dir: ./outputs",
                        "output_name: \"\"",
                        f"run_name: {run_name}",
                        "checkpoint_dir: ./checkpoints",
                    ]
                )
            )
            metrics_path = root / "outputs" / run_name / "metrics.csv"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,phase,eval_top1_error,val_top1_error,best_top1_error,test_top1_error",
                        "200,train_val,0.399100,0.399100,0.399100,",
                        "200,final_test,,,0.399100,0.405600",
                    ]
                )
            )
            best_path = root / "checkpoints" / run_name / "best.pt"
            best_path.parent.mkdir(parents=True)
            best_path.write_bytes(b"checkpoint payload is not loaded when sidecar metadata is valid")
            raw_config = load_config(str(root / config_path))
            best_path.with_suffix(best_path.suffix + ".json").write_text(
                json.dumps(
                    {
                        "epoch": 197,
                        "best_acc": 60.09,
                        "best_epoch": 197,
                        "config": expected_resume_config(raw_config),
                    }
                )
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "allthemix.cli.summarize",
                    "--root",
                    str(root),
                    "--preset",
                    "tiny-imagenet-xla4",
                    "--format",
                    "commands",
                    "--method",
                    "baseline",
                    "--include-complete",
                    "--train-arg=--eval-only",
                    "--device",
                    "xla",
                    "--num-cores",
                    "4",
                    "--num-workers",
                    "0",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = result.stdout.strip().splitlines()
        self.assertEqual(len(commands), 1)
        self.assertIn("run_tiny_imagenet_preact_resnet18_baseline_xla4.sh", commands[0])
        self.assertIn("--eval-only", commands[0])
        self.assertIn(f"--checkpoint {best_path.as_posix()}", commands[0])

    def test_readme_manual_xla4_commands_match_generated_full_commands(self):
        readme = Path("README.md").read_text()
        commands = render_commands(
            summarize_experiments(Path("."), TINY_IMAGENET_XLA4_SPECS),
            device="xla",
            num_cores=4,
            num_workers=0,
        ).splitlines()

        positions = []
        for command in commands:
            with self.subTest(command=command):
                self.assertIn(command, readme)
                positions.append(readme.index(command))
        self.assertEqual(positions, sorted(positions))

    def test_tiny_imagenet_xla4_suite_excludes_legacy_paper_configs(self):
        commands = render_commands(
            summarize_experiments(Path("."), TINY_IMAGENET_XLA4_SPECS),
            device="xla",
            num_cores=4,
            num_workers=0,
        )

        self.assertIn("run_tiny_imagenet_preact_resnet18_mixup_xla4.sh", commands)
        self.assertIn("run_tiny_imagenet_preact_resnet18_fmix_xla4.sh", commands)
        self.assertNotIn("paper_xla4", commands)
        for spec in TINY_IMAGENET_XLA4_SPECS:
            with self.subTest(method=spec.method_key):
                self.assertNotIn("paper_xla4", spec.config_path.as_posix())
                self.assertNotIn("paper_xla4", script_path_for_spec(spec).as_posix())

    def test_readme_documents_guided_sr_greedy_pairing_protocol(self):
        readme = Path("README.md").read_text()
        normalized = " ".join(readme.split())

        self.assertIn("Guided-SR follows the GuidedMixup SR-style greedy pairing setting", normalized)
        self.assertIn("`cross_device_shuffle` is disabled for that config", normalized)
        self.assertIn("scope=(0.1, 0.4)", normalized)

    def test_readme_training_flow_mentions_all_mixda_methods_generically(self):
        readme = Path("README.md").read_text()
        normalized = " ".join(readme.split())

        self.assertIn("data -> basic aug/preprocess -> batch -> optional MixDA method -> train loop", normalized)
        self.assertNotIn("optional FMix/MixUp method", normalized)
        for method in ("cutmix", "resizemix", "saliencymix", "guided_sr", "catchupmix"):
            with self.subTest(method=method):
                self.assertIn(f"`method: {method}`", readme)

    def test_readme_documents_checkpoint_dir_auto_checkpoint_safety(self):
        readme = Path("README.md").read_text()
        normalized = " ".join(readme.split())

        self.assertIn("Use `--train-arg=--checkpoint-dir` for temporary smoke/debug checkpoint roots", normalized)
        self.assertIn("`./checkpoints/tiny_imagenet_xla4_smoke`", readme)
        self.assertIn("disable automatic resume and automatic eval-only best-checkpoint loading", normalized)
        self.assertIn("unless you also pass an explicit `--checkpoint`", normalized)

    def test_readme_collect_require_complete_matches_script_gate_order(self):
        readme = Path("README.md").read_text()
        normalized = " ".join(readme.split())
        script = Path("scripts/experiment_run/collect_tiny_imagenet_xla4_results.sh").read_text()

        self.assertLess(
            script.index('if [[ "$require_complete" == true ]]; then'),
            script.index('tee "$summary_dir/default.csv"'),
        )
        self.assertIn("the completeness gate runs before writing the final table artifacts", normalized)
        self.assertIn("removes stale final table files from earlier collects", normalized)
        self.assertIn("`collect_manifest.json`, `protocol.txt`, and `status.md` remain available", normalized)
        self.assertNotIn("artifacts are still written first", normalized)

    def test_readme_documents_final_test_checkpoint_summary_fields(self):
        readme = Path("README.md").read_text()
        normalized = " ".join(readme.split())

        self.assertIn("`final_test_checkpoint`", readme)
        self.assertIn("`final_test_checkpoint_source`", readme)
        self.assertIn("final-test checkpoint policy", normalized)
        self.assertIn("final-test checkpoint source", normalized)

    def test_tiny_imagenet_xla4_suite_manifest_python_is_syntax_valid(self):
        script = Path("scripts/experiment_run/run_tiny_imagenet_xla4_suite.sh").read_text()
        start_marker = "<<'PY'\n"
        end_marker = "\nPY\n\npython -m allthemix.cli.summarize"
        manifest_code = script.split(start_marker, 1)[1].split(end_marker, 1)[0]

        compile(manifest_code, "run_tiny_imagenet_xla4_suite_manifest.py", "exec")

    def test_tiny_imagenet_xla4_collect_manifest_python_is_syntax_valid(self):
        script = Path("scripts/experiment_run/collect_tiny_imagenet_xla4_results.sh").read_text()
        start_marker = "<<'PY'\n"
        end_marker = "\nPY\n\npython -m allthemix.cli.summarize"
        manifest_code = script.split(start_marker, 1)[1].split(end_marker, 1)[0]

        compile(manifest_code, "collect_tiny_imagenet_xla4_manifest.py", "exec")


if __name__ == "__main__":
    unittest.main()
