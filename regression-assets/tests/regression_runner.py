from __future__ import annotations

import getpass
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.regression_config import RegressionConfig


@dataclass(frozen=True)
class CommandStep:
    name: str
    command: str | Sequence[str]
    log_name: str
    cwd: Path | str | None = None
    env: Mapping[str, str] | None = None
    critical: bool = True
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class StepResult:
    name: str
    command: str
    cwd: str
    log_path: str
    exit_code: int
    duration_seconds: float
    started_at: str
    finished_at: str
    critical: bool

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def status(self) -> str:
        if self.passed:
            return 'PASS'
        if self.critical:
            return 'FAIL'
        return 'NON_BLOCKING_FAIL'


class RegressionRunner:
    def __init__(self, config: RegressionConfig) -> None:
        self.config = config
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[StepResult] = []
        self.environment_info: dict[str, Any] = {}

    def run(self, step: CommandStep) -> StepResult:
        command_text = self._command_to_text(step.command)
        cwd = Path(step.cwd).resolve() if step.cwd else self.config.repo_root
        log_path = self.config.log_dir / step.log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_dir = self.config.log_dir / 'tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                'REGRESSION_LOG_DIR': str(self.config.log_dir),
                'PYTHONUNBUFFERED': '1',
                'TMPDIR': str(tmp_dir),
                'TEMP': str(tmp_dir),
                'TMP': str(tmp_dir),
                'GOTMPDIR': str(tmp_dir),
            }
        )
        if step.env:
            env.update(step.env)

        started = datetime.now(timezone.utc)
        monotonic_start = time.monotonic()
        exit_code = 127

        separator = '=' * 80
        footer = '-' * 80
        with log_path.open('a', encoding='utf-8') as log_file:
            log_file.write(f'{separator}\n')
            log_file.write(f'Step: {step.name}\n')
            log_file.write(f'Started UTC: {started.isoformat()}\n')
            log_file.write(f'CWD: {cwd}\n')
            log_file.write(f'Command: {command_text}\n')
            log_file.write(f'{separator}\n\n')
            log_file.flush()

            try:
                completed = subprocess.run(
                    step.command,
                    cwd=cwd,
                    env=env,
                    shell=isinstance(step.command, str),
                    executable='/bin/bash' if isinstance(step.command, str) else None,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=step.timeout_seconds,
                    check=False,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired as exc:
                exit_code = 124
                log_file.write(f'\n# TIMEOUT after {exc.timeout} seconds\n')
            except FileNotFoundError as exc:
                exit_code = 127
                log_file.write(f'\n# COMMAND NOT FOUND: {exc}\n')

            finished = datetime.now(timezone.utc)
            duration = time.monotonic() - monotonic_start
            log_file.write('\n')
            log_file.write(f'{footer}\n')
            log_file.write(f'Finished UTC: {finished.isoformat()}\n')
            log_file.write(f'Exit code: {exit_code}\n')
            log_file.write(f'Duration seconds: {duration:.3f}\n')
            log_file.write(f'{footer}\n\n')

        result = StepResult(
            name=step.name,
            command=command_text,
            cwd=str(cwd),
            log_path=str(log_path),
            exit_code=exit_code,
            duration_seconds=duration,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            critical=step.critical,
        )
        self.results.append(result)
        return result

    def run_many(self, steps: Sequence[CommandStep]) -> list[StepResult]:
        results: list[StepResult] = []
        failures: list[StepResult] = []

        for step in steps:
            result = self.run(step)
            results.append(result)
            if not result.passed and result.critical:
                failures.append(result)
                if not self.config.continue_on_error:
                    break

        if failures:
            details = '\n'.join(
                f'- {failure.name}: exit {failure.exit_code}, log {failure.log_path}' for failure in failures
            )
            raise AssertionError(f'Regression command failures:\n{details}')

        return results

    def results_for(self, names: Sequence[str]) -> list[StepResult]:
        wanted = set(names)
        return [result for result in self.results if result.name in wanted]

    def collect_environment_report_info(self) -> None:
        self.collect_commit_report_info()
        self.environment_info.update(
            {
                'tester': getpass.getuser() or os.environ.get('USER') or os.environ.get('USERNAME') or 'N/A',
                'module_versions': [
                    {'module': 'gcc', 'version': self._first_line(['gcc', '--version'])},
                    {'module': 'g++', 'version': self._first_line(['g++', '--version'])},
                    {'module': 'cmake', 'version': self._first_line(['cmake', '--version'])},
                    {'module': 'go', 'version': self._first_line(['go', 'version'])},
                    {'module': 'python3', 'version': self._first_line(['python3', '--version'])},
                    {'module': 'docker', 'version': self._first_line(['docker', '--version'])},
                ],
            }
        )

    def collect_commit_report_info(self) -> None:
        commits = self.environment_info.get('commits', {})
        if not isinstance(commits, dict):
            commits = {}
        commits.update(self._collect_submodule_commits())
        self.environment_info['commits'] = commits

    def collect_lattisense_report_info(self) -> None:
        self.collect_commit_report_info()

    def _collect_submodule_commits(self) -> dict[str, str]:
        repo_root = self.config.repo_root
        return {
            'lattisense': self._git_head(repo_root / 'inference' / 'lattisense'),
            'HEonGPU': self._git_head(repo_root / 'inference' / 'lattisense' / 'HEonGPU'),
            'Lattigo': self._git_head(repo_root / 'inference' / 'lattisense' / 'fhe_ops_lib' / 'lattigo'),
        }

    @staticmethod
    def _first_line(command: Sequence[str]) -> str:
        executable = command[0]
        if shutil.which(executable) is None:
            return 'N/A'
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return 'N/A'
        for line in completed.stdout.splitlines():
            if line.strip():
                return line.strip()
        return 'N/A'

    @staticmethod
    def _git_head(path: Path) -> str:
        if not path.exists():
            return 'N/A'
        try:
            completed = subprocess.run(
                ['git', '-C', str(path), 'rev-parse', 'HEAD'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return 'N/A'
        if completed.returncode != 0:
            return 'N/A'
        return completed.stdout.strip() or 'N/A'

    @staticmethod
    def _command_to_text(command: str | Sequence[str]) -> str:
        if isinstance(command, str):
            return command
        return ' '.join(shlex.quote(part) for part in command)
