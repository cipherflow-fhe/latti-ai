from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tests import regression_config
from tests.regression_config import build_config


class FixedDatetime:
    @staticmethod
    def now() -> datetime:
        return datetime(2026, 7, 24, 15, 30, 45)


def _build(tmp_path: Path, *, log_dir: str | None = None):
    return build_config(
        repo_root=str(tmp_path),
        log_dir=log_dir,
        scope='environment',
        cuda_arch=None,
        continue_on_error=False,
        python_bin='python3',
        install_python_deps=False,
        baseline=None,
        target='target-commit',
        skip_performance=False,
        skip_docker=False,
        skip_lattisense=False,
    )


def test_default_log_dir_groups_runs_by_date_then_time(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(regression_config, 'datetime', FixedDatetime)

    config = _build(tmp_path)

    assert config.log_dir == tmp_path / 'logs' / '20260724' / '153045'


def test_explicit_log_dir_is_preserved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(regression_config, 'datetime', FixedDatetime)
    explicit_log_dir = tmp_path / 'custom-logs' / 'manual-run'

    config = _build(tmp_path, log_dir=str(explicit_log_dir))

    assert config.log_dir == explicit_log_dir.resolve()