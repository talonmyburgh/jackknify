import typer
from .realise import realise
from .noise import noise
from .make_test_ms import make_ms

app = typer.Typer(
    name="jackknify",
    help="Jackknife interferometric datasets using JAX.",
    no_args_is_help=True,
)

app.command(name="realise")(realise)
app.command(name="noise")(noise)
app.command(name="make-test-ms")(make_ms)