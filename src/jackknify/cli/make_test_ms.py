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
    ms_file: Annotated[
        MS, 
        typer.Argument(help="Path to create the mock MS.")
    ],
    rows: Annotated[int, typer.Option()] = 100,
    chans: Annotated[int, typer.Option()] = 16,
):
    from jackknify.core.MSHandler import MSWrapper
    try:
        MSWrapper.create_test_ms(str(ms_file), n_rows=rows, n_chan=chans)
        print("Test MS created successfully.")
    except Exception as e:
        print(f"Error creating MS: {e}")