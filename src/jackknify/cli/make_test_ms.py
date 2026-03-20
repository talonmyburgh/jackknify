from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo.utils.decorators import stimela_cab

MS = NewType("MS", Path)


@stimela_cab(
    name="make_ms",
    info="Creates a simple mock MS filled with 1s for testing.",
)
def make_ms(
    ms_file: Annotated[MS, typer.Argument(..., parser=Path, help="Path to create the mock MS.")],
    rows: Annotated[int, typer.Option()] = 100,
    chans: Annotated[int, typer.Option()] = 16,
):
    """
    Creates a simple mock MS filled with 1s for testing.
    """
    from jackknify.core.make_test_ms import make_ms as make_ms_core

    make_ms_core(ms_file=str(ms_file), rows=rows, chans=chans)
