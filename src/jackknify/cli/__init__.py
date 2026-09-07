"""CLI for jackknify."""

import typer

app = typer.Typer(
    name="jackknify",
    help="Jackknife interferometric datasets using JAX.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Jackknife interferometric datasets using JAX."""
    pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from jackknify.cli.make_ms import make_ms  # noqa: E402
from jackknify.cli.noise import noise  # noqa: E402
from jackknify.cli.onboard import onboard  # noqa: E402
from jackknify.cli.realise import realise  # noqa: E402

app.command(name="realise")(realise)
app.command(name="noise")(noise)
app.command(name="make-ms")(make_ms)
app.command(name="onboard")(onboard)

__all__ = ["app"]
