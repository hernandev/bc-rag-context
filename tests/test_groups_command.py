import json
import stat
from pathlib import Path

import pytest

from bc_rag.config import load_config


def _write_config(root: Path, command: str | list[str]) -> None:
    (root / ".bc-rag.json").write_text(
        json.dumps(
            {
                "defaultSpace": "prose",
                "groups": [
                    {"name": "static-docs", "include": ["docs/**/*.md"], "space": "prose"}
                ],
                "spaces": {
                    "prose": {
                        "dense": {
                            "provider": "local",
                            "model": "jinaai/jina-embeddings-v2-base-en",
                        },
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    },
                    "code": {
                        "dense": {"provider": "voyage", "model": "voyage-code-4"},
                        "sparse": {"provider": "local", "model": "Qdrant/bm25"},
                        "rerank": {"provider": "jina", "model": "jina-reranker-v3.5"},
                        "dimensions": 1024,
                        "chunk": {"max_chars": 2400, "min_chars": 40},
                    },
                },
                "groupsCommand": command,
            }
        ),
        encoding="utf-8",
    )


def test_top_level_tags_are_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "groups.py")
    file_data = json.loads((tmp_path / ".bc-rag.json").read_text())
    file_data["tags"] = ["scope:internal"]
    (tmp_path / ".bc-rag.json").write_text(json.dumps(file_data), encoding="utf-8")
    with pytest.raises(ValueError, match="tags is not a config key. Set facets on each group"):
        load_config(tmp_path)


def test_groups_false_skips_a_failing_command(tmp_path: Path) -> None:
    _write_config(tmp_path, ["/no/such/program"])
    with pytest.raises(ValueError, match="groupsCommand could not start"):
        load_config(tmp_path)
    configuration, _path = load_config(tmp_path, groups=False)
    assert configuration.is_partial()
    assert [group.name for group in configuration.groups] == ["static-docs"]


def test_a_partial_config_cannot_list_files(tmp_path: Path) -> None:
    from bc_rag.discover import iter_source_files, source_for_path

    _write_config(tmp_path, ["/no/such/program"])
    configuration, _path = load_config(tmp_path, groups=False)
    with pytest.raises(ValueError, match="without groupsCommand"):
        list(iter_source_files(tmp_path, configuration))
    with pytest.raises(ValueError, match="without groupsCommand"):
        source_for_path("docs/a.md", tmp_path / "docs" / "a.md", configuration)


def test_groups_command_appends_json_groups(tmp_path: Path) -> None:
    script = tmp_path / "groups.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('{\"groups\": [{\"name\": \"from-command\", "
        "\"include\": [\"libs/**/*.py\"], \"space\": \"code\"}]}')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    _write_config(tmp_path, "groups.py")
    configuration, _path = load_config(tmp_path)
    names = [group.name for group in configuration.groups]
    assert names == ["static-docs", "from-command"]
    assert configuration.groups[1].space == "code"


def test_groups_command_rejects_a_repeated_name(tmp_path: Path) -> None:
    script = tmp_path / "groups.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('{\"groups\": [{\"name\": \"static-docs\", \"space\": \"prose\", "
        "\"include\": [\"libs/**/*.py\"]}]}')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    _write_config(tmp_path, "groups.py")
    with pytest.raises(ValueError, match="repeated the group name 'static-docs'"):
        load_config(tmp_path)


def test_sources_groups_lists_static_and_command_groups(tmp_path: Path) -> None:
    script = tmp_path / "groups.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('{\"groups\": [{\"name\": \"from-command\", "
        "\"include\": [\"libs/**/*.py\"], \"space\": \"code\"}]}')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    _write_config(tmp_path, "groups.py")
    from typer.testing import CliRunner

    from bc_rag.catalog import register_project
    from bc_rag.cli import app

    register_project(tmp_path)
    result = CliRunner().invoke(app, ["sources", "groups", "--root", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    names = {group["name"] for group in payload["groups"]}
    assert names == {"static-docs", "from-command"}


def test_groups_command_rejects_stdout_that_is_not_groups(tmp_path: Path) -> None:
    script = tmp_path / "groups.py"
    script.write_text("#!/usr/bin/env python3\nprint('[]')\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    _write_config(tmp_path, ["python3", "groups.py"])
    with pytest.raises(ValueError, match="groups array"):
        load_config(tmp_path)
