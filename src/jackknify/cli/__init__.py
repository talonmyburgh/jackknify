import typer

from jackknify.cli.make_test_ms import make_ms
from jackknify.cli.noise import noise
from jackknify.cli.onboard import onboard
from jackknify.cli.realise import realise

app = typer.Typer(
    name="jackknify",
    help="Jackknife interferometric datasets using JAX.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Jackknife interferometric datasets using JAX."""
    pass


app.command(name="realise")(realise)
app.command(name="noise")(noise)
app.command(name="make-test-ms")(make_ms)
app.command(name="onboard")(onboard)

__all__ = ["app"]
