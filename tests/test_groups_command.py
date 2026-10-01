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
                        "dense": {"provider": "local", "model": "jinaai/jina-embeddings-v2-base-en"},
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
    with pytest.raises(ValueError, match="tags is not a top-level key"):
        load_config(tmp_path)


def test_groups_command_appends_json_groups(tmp_path: Path) -> None:
    script = tmp_path / "groups.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('{\"groups\": [{\"name\": \"from-command\", \"include\": [\"libs/**/*.py\"], \"space\": \"code\"}]}')\n",
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
        "print('{\"groups\": [{\"name\": \"static-docs\", \"include\": [\"libs/**/*.py\"]}]}')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    _write_config(tmp_path, "groups.py")
    with pytest.raises(ValueError, match="static-docs"):
        load_config(tmp_path)


def test_config_command_prints_static_and_command_groups(tmp_path: Path) -> None:
    script = tmp_path / "groups.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('{\"groups\": [{\"name\": \"from-command\", \"include\": [\"libs/**/*.py\"], \"space\": \"code\"}]}')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    _write_config(tmp_path, "groups.py")
    from typer.testing import CliRunner

    from bc_rag.cli import app

    result = CliRunner().invoke(app, ["config", "dump", "--root", str(tmp_path)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    names = [group["name"] for group in payload["groups"]]
    assert names == ["static-docs", "from-command"]
    assert list(payload)[-1] == "groups"
    assert payload["groupsCommand"] == "groups.py"
    assert payload["defaultSpace"] == "prose"
    assert "embed" not in payload
    assert "chunk" not in payload
    assert "openapi" not in payload


def test_groups_command_rejects_stdout_that_is_not_groups(tmp_path: Path) -> None:
    script = tmp_path / "groups.py"
    script.write_text("#!/usr/bin/env python3\nprint('[]')\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    _write_config(tmp_path, ["python3", "groups.py"])
    with pytest.raises(ValueError, match="groups array"):
        load_config(tmp_path)
