from typing import Annotated

import typer
from hip_cargo.utils.decorators import stimela_cab, stimela_output


@stimela_cab(
    name="make_ms",
    info="Creates a simple mock MS filled with 1s for testing.",
)
@stimela_output(
    name="ms_file",
    dtype="MS",
    info="The resulting mock Measurement Set.",
)
def make_ms(
    # FIX: Add the ... back into typer.Argument
    ms_file: Annotated[str, typer.Argument(..., help="Path to create the mock MS.")],
    rows: Annotated[int, typer.Option()] = 100,
    chans: Annotated[int, typer.Option()] = 16,
):
    """
    Creates a simple mock MS filled with 1s for testing.
    """
    from jackknify.core.make_test_ms import make_ms as make_ms_core

    make_ms_core(ms_file=ms_file, rows=rows, chans=chans)
