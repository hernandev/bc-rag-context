"""bc-rag command line.

Verbs at the top work on one project's index (`index`, `search`, `status`, `facets`).
Nouns group the rest: `sources`, `project`, `config`, `models`, `services`. A project
is named with --project, else it is the one that holds the current folder.
"""

from __future__ import annotations

from bc_rag.cli import index as _index  # noqa: F401  registers index, search, status, facets
from bc_rag.cli._app import app
from bc_rag.cli.config import config_app
from bc_rag.cli.model_cache import models_app
from bc_rag.cli.project import project_app
from bc_rag.cli.services import services_app
from bc_rag.cli.sources import sources_app

# root help lists the groups in this order, each under its own heading.
app.add_typer(sources_app, name="sources", rich_help_panel="Sources")
app.add_typer(project_app, name="project", rich_help_panel="Projects")
app.add_typer(config_app, name="config", rich_help_panel="Config")
app.add_typer(models_app, name="models", rich_help_panel="Models")
app.add_typer(services_app, name="services", rich_help_panel="Services")

__all__ = ["app"]
