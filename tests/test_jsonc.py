from pathlib import Path

from bc_rag.config import load_config


def test_load_config_allows_line_comments(tmp_path: Path) -> None:
    (tmp_path / ".bc-rag.json").write_text(
        """
{
  "groups": [
    { "name": "docs-bigcolony-reference", "include": ["docs/**/*.md"], "priority": 80 }
    // {
    //   "name": "libs-testing-mock-core-testing-mock-core-faker",
    //   "include": ["libs/testing/**/src/**/*.ts"],
    //   "priority": 51
    // }
  ]
}
""",
        encoding="utf-8",
    )
    configuration, _ = load_config(tmp_path)
    names = [group.name for group in configuration.resolved_groups()]
    assert names == ["docs-bigcolony-reference"]
