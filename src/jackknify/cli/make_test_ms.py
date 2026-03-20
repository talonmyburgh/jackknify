from typing import Annotated

import typer
from hip_cargo.utils.decorators import stimela_cab, stimela_output


@stimela_cab(
    name="make_ms",
    info="Creates a simple mock MS filled with 1s for testing.",
)
@stimela_output(
    name="out_ms",  # Unique name so it doesn't conflict with the input
    dtype="MS",
    info="The resulting mock Measurement Set.",
    implicit="{ms-file}",  # This links the output path to the ms-file input!
)
def make_ms(
    ms_file: Annotated[str, typer.Argument(..., help="Path to create the mock MS.")],
    rows: Annotated[int, typer.Option()] = 100,
    chans: Annotated[int, typer.Option()] = 16,
):
    """
    Creates a simple mock MS filled with 1s for testing.
    """
    from jackknify.core.make_test_ms import make_ms as make_ms_core

    make_ms_core(ms_file=ms_file, rows=rows, chans=chans)
