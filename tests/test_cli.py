import json
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bc_rag import catalog
from bc_rag.cli import app
from bc_rag.usersettings import load_settings, settings_path

runner = CliRunner()

TOP = ["index", "search", "status", "facets"]
LEAVES = [
    "sources files",
    "sources groups",
    "sources chunks",
    "project list",
    "project add",
    "project clear",
    "project remove",
    "config list",
    "config get",
    "config set",
    "config unset",
    "models list",
    "models add",
    "models remove",
]
GONE = ["settings", "project index", "project files", "project config", "project show"]


@pytest.fixture(autouse=True)
def wide_consoles(monkeypatch: pytest.MonkeyPatch) -> None:
    # long temporary paths must not wrap or be cut by Rich.
    from bc_rag.cli import _app

    monkeypatch.setattr(_app.console, "width", 400)
    monkeypatch.setattr(_app.err, "width", 400)


def _run(*args: str):
    return runner.invoke(app, list(args))


def _added(tmp_path: Path, name: str = "repo") -> Path:
    root = tmp_path / name
    root.mkdir()
    result = _run("project", "add", "--root", str(root))
    assert result.exit_code == 0, result.output
    return root


def test_root_help_lists_every_command_under_its_heading() -> None:
    result = _run("--help")
    assert result.exit_code == 0
    for name in [*TOP, *LEAVES, "sources", "project", "config", "models", "services"]:
        assert name in result.output, name
    for heading in ("Index", "Sources", "Projects", "Config", "Models", "Services"):
        assert f"─ {heading} ─" in result.output, heading
    # no empty default box left over from the headings.
    assert "─ Commands ─" not in result.output


@pytest.mark.parametrize("name", GONE)
def test_old_commands_are_gone(name: str) -> None:
    assert _run(*name.split(), "--help").exit_code == 2


@pytest.mark.parametrize("group", ["sources", "project", "config", "models"])
def test_bare_group_exits_2_and_names_its_subcommands(group: str) -> None:
    result = _run(group)
    assert result.exit_code == 2
    assert "Please specify a subcommand" in result.output
    for leaf in LEAVES:
        if leaf.startswith(f"{group} "):
            assert leaf.split(" ", 1)[1] in result.output


def test_project_add_writes_config_and_registers(tmp_path: Path) -> None:
    root = _added(tmp_path)
    assert (root / ".bc-rag.json").is_file()
    assert [entry.name for entry in catalog.load_catalog()] == ["repo"]
    again = _run("project", "add", "--root", str(root))
    assert again.exit_code == 1
    assert "already registered as repo" in again.output


def test_project_add_keeps_an_existing_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    from bc_rag.config import dump_default_config

    text = dump_default_config().replace('"default"', '"main"')
    (root / ".bc-rag.json").write_text(text, encoding="utf-8")
    result = _run("project", "add", "--root", str(root))
    assert result.exit_code == 0, result.output
    assert "(kept)" in result.output
    assert (root / ".bc-rag.json").read_text(encoding="utf-8") == text


def test_project_add_refuses_a_broken_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".bc-rag.json").write_text("{}", encoding="utf-8")
    result = _run("project", "add", "--root", str(root))
    assert result.exit_code == 1
    assert catalog.load_catalog() == []


def test_project_add_names_the_project_with_the_project_flag(tmp_path: Path) -> None:
    _added(tmp_path, "one")
    other = tmp_path / "two"
    other.mkdir()
    taken = _run("project", "add", "--root", str(other), "--project", "one")
    assert taken.exit_code == 1
    assert "name 'one' is used by" in taken.output
    named = _run("project", "add", "--root", str(other), "-p", "second")
    assert named.exit_code == 0, named.output
    assert "second" in {entry.name for entry in catalog.load_catalog()}


def test_resolution_by_name_from_an_unrelated_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _added(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = _run("config", "get", "every", "-p", "repo")
    # not set yet: exit 1, but the project was found.
    assert result.exit_code == 1, result.output
    assert "not set" in result.output


def test_resolution_walks_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _added(tmp_path)
    deep = root / "libs" / "x"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    result = _run("sources", "groups", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["project"] == "repo"


def test_outside_a_project_exits_2_and_lists_the_names(tmp_path: Path) -> None:
    _added(tmp_path, "repo")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = _run("sources", "groups", "--root", str(elsewhere))
    assert result.exit_code == 2
    assert f"not a bc-rag project: {elsewhere.resolve()}" in result.output
    assert "pass --project NAME. Registered: repo" in result.output
    assert "bc-rag project add" in result.output
    unknown = _run("sources", "groups", "-p", "nope")
    assert unknown.exit_code == 2
    assert "unknown project: nope" in unknown.output


def test_project_and_root_together_exit_2(tmp_path: Path) -> None:
    result = _run("sources", "groups", "-p", "x", "--root", str(tmp_path))
    assert result.exit_code == 2
    assert "use one of --project or --root" in result.output


def test_config_defaults_are_written_with_owner_only_mode() -> None:
    result = _run("config", "list", "--global")
    assert result.exit_code == 0, result.output
    assert load_settings() == {
        "autostart": "true",
        "docker-context": "",
        "qdrant": "docker",
        "qdrant-url": "http://127.0.0.1:32321",
    }
    assert stat.S_IMODE(settings_path().stat().st_mode) == 0o600
    assert "voyage-api-key" in result.output
    assert "every" in result.output


@pytest.mark.parametrize(("key", "value"), [("autostart", "maybe"), ("qdrant", "both")])
def test_config_set_rejects_bad_values(key: str, value: str) -> None:
    result = _run("config", "set", key, value, "--global")
    assert result.exit_code == 1


def test_a_global_only_key_needs_the_global_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _added(tmp_path)
    monkeypatch.chdir(root)
    result = _run("config", "set", "autostart", "false")
    assert result.exit_code == 2
    assert "autostart applies to every project. Add --global" in result.output
    assert _run("config", "set", "autostart", "false", "--global").exit_code == 0
    assert load_settings()["autostart"] == "false"


def test_every_is_set_for_this_project_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _added(tmp_path)
    monkeypatch.chdir(root)
    assert _run("config", "set", "every", "10m").exit_code == 0
    assert catalog.load_catalog()[0].every == "10m"
    assert "every" not in load_settings()
    got = _run("config", "get", "every")
    assert got.stdout.strip() == "10m"
    assert "you, this project" in got.output
    assert _run("config", "set", "every", "off").exit_code == 0
    assert catalog.load_catalog()[0].every == "off"
    assert _run("config", "unset", "every").exit_code == 0
    assert catalog.load_catalog()[0].every is None
    assert _run("config", "set", "every", "soon").exit_code == 1
    assert _run("config", "set", "interval", "5m").exit_code == 2


def test_a_global_every_reaches_a_project_without_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _added(tmp_path)
    monkeypatch.chdir(root)
    assert _run("config", "set", "every", "30m", "--global").exit_code == 0
    assert load_settings()["every"] == "30m"
    got = _run("config", "get", "every")
    assert got.stdout.strip() == "30m"
    assert "you, every project" in got.output
    assert _run("config", "set", "every", "5m").exit_code == 0
    assert _run("config", "get", "every").stdout.strip() == "5m"
    assert _run("config", "get", "every", "--global").stdout.strip() == "30m"


def test_config_set_outside_a_project_needs_one(tmp_path: Path, monkeypatch) -> None:
    _added(tmp_path, "repo")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = _run("config", "set", "every", "10m")
    assert result.exit_code == 2
    assert "not inside a project. Pass --project NAME (repo), or --global." in result.output
    assert _run("config", "set", "every", "10m", "-p", "repo").exit_code == 0
    assert _run("config", "set", "every", "10m", "--global", "-p", "repo").exit_code == 2


def test_project_remove_keeps_the_folder(tmp_path: Path) -> None:
    root = _added(tmp_path)
    result = _run("project", "remove", "--root", str(root))
    assert result.exit_code == 0, result.output
    assert catalog.load_catalog() == []
    assert (root / ".bc-rag.json").is_file()
    assert "bc-rag project clear" in result.output


def test_sources_groups_filters_by_facet(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    raw = json.loads((Path(__file__).parent / "fixtures_config.json").read_text())
    (root / ".bc-rag.json").write_text(json.dumps(raw), encoding="utf-8")
    assert _run("project", "add", "--root", str(root)).exit_code == 0

    def names(*flags: str) -> list[str]:
        result = _run("sources", "groups", "--root", str(root), "--json", *flags)
        assert result.exit_code == 0, result.output
        return [group["name"] for group in json.loads(result.stdout)["groups"]]

    assert names() == ["olo-spec", "engine-docs", "admin-docs"]
    assert names("--facet", "area=engine") == ["engine-docs"]
    assert names("-f", "area=engine", "-f", "area=admin") == ["engine-docs", "admin-docs"]
    assert names("--exclude", "scope=external") == ["engine-docs", "admin-docs"]
    assert names("--facet", "group=olo-spec") == ["olo-spec"]
    bad = _run("sources", "groups", "--root", str(root), "--facet", "area")
    assert bad.exit_code == 2
    assert "key=value" in bad.output
    timing = _run("sources", "groups", "--root", str(root), "--timing", "--json")
    assert timing.exit_code == 0, timing.output
    assert "total_glob_ms" in json.loads(timing.stdout)


def test_index_flags_that_need_watch(tmp_path: Path) -> None:
    root = _added(tmp_path)
    for flags in (["--skip-initial"], ["--debounce", "1"]):
        result = _run("index", "--root", str(root), *flags)
        assert result.exit_code == 2
        assert "need --watch" in result.output
    both = _run("index", "--root", str(root), "--watch", "--path", "a.ts")
    assert both.exit_code == 2


def test_help_never_brings_up_the_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr("bc_rag.services.ensure_stack", lambda **kwargs: calls.append("stack"))
    assert _run("index", "--help").exit_code == 0
    assert calls == []
    root = _added(tmp_path)

    class Stats:
        errors: list[str] = []
        stopped = False
        scanned = indexed_files = skipped_unchanged = skipped_large = 0
        deleted_files = chunks = jina_embed_calls = jina_embed_tokens = 0

    monkeypatch.setattr("bc_rag.runtime.index_project", lambda entry, **kwargs: Stats())
    result = _run("index", "--root", str(root))
    assert result.exit_code == 0, result.output
    assert calls == ["stack"]
