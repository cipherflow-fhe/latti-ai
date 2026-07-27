from __future__ import annotations

from pathlib import Path

import pytest

from tests.regression_config import RegressionConfig


REQUIRED_DOC_AND_CONFIG_PATHS = [
    'README.md',
    'docs/en/build-guide.md',
    'docs/en/APIs_Reference.md',
    'examples/README.md',
    'inference/lattisense/README.md',
    'inference/lattisense/README_zh.md',
    'Dockerfile',
]


@pytest.mark.regression_scope('docs')
def test_release_documentation_inputs_exist(regression_config: RegressionConfig) -> None:
    missing = [
        relative_path
        for relative_path in REQUIRED_DOC_AND_CONFIG_PATHS
        if not (regression_config.repo_root / relative_path).exists()
    ]
    assert not missing, 'Missing release documentation inputs: ' + ', '.join(missing)


@pytest.mark.regression_scope('docs')
def test_write_documentation_consistency_checklist(regression_config: RegressionConfig) -> None:
    regression_config.log_dir.mkdir(parents=True, exist_ok=True)
    checklist_path = regression_config.log_dir / 'docs-consistency-checklist.md'
    checklist_path.write_text(_build_checklist(regression_config.repo_root), encoding='utf-8')
    assert checklist_path.exists()


def _build_checklist(repo_root: Path) -> str:
    rows = [
        ('CHANGELOG 覆盖本次用户可见变更', '需人工依据合并范围确认'),
        ('README 构建和快速开始准确', '检查 README.md'),
        ('Build guide 构建选项准确', '检查 docs/en/build-guide.md'),
        ('API 文档与代码一致', '检查 docs/en/APIs_Reference.md'),
        ('示例 README 与实际命令一致', '检查 examples/README.md'),
        ('lattisense 英文文档同步', '检查 inference/lattisense/README.md'),
        ('lattisense 中文文档同步需求已评估', '检查 inference/lattisense/README_zh.md'),
        ('Dockerfile 与构建文档一致', '检查 Dockerfile'),
    ]

    lines = [
        '# CHANGELOG 与文档一致性检查清单',
        '',
        f'仓库路径：`{repo_root}`',
        '',
        '| 检查项 | 结果 | 说明 | 关联 PR / commit |',
        '| --- | --- | --- | --- |',
    ]
    for item, note in rows:
        lines.append(f'| {item} |  | {note} |  |')
    return '\n'.join(lines) + '\n'
