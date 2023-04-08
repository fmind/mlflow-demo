"""Tasks of the project."""

# %% IMPORTS

from invoke import task
from invoke.context import Context

# %% TASKS

@task
def install(c: Context) -> None:
    """Install the project."""
    c.run("poetry install")


@task
def serve(c: Context) -> None:
    """Tear up the project docker."""
    c.run("docker-compose up")
