import typer
from .realize import realize
from .noise import noise
from .make_ms import make_ms

app = typer.Typer(
    name="jackknify",
    help="Jackknife interferometric datasets using JAX.",
    no_args_is_help=True,
)

app.command(name="realize")(realize)
app.command(name="noise")(noise)
app.command(name="make-ms")(make_ms)