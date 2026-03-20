from pathlib import Path
from typing import Annotated, NewType, Optional

import typer
from hip_cargo.utils.decorators import stimela_cab, stimela_output

# Define types for Stimela type inference
MS = NewType("MS", Path)
Directory = NewType("Directory", Path)

@stimela_cab(
    name="realise",
    info="Generates jackknife noise realisations from a Measurement Set.",
)
@stimela_output(
    name="out_dir",
    dtype="Directory",
    required=False,
    info="Output directory (only used if mode is 'copy').",
)
def realise(
    ms_file: Annotated[MS, typer.Argument(..., parser=Path, help="Input Measurement Set.")],
    col: Annotated[str, typer.Option(help="Input data column name.")] = "DATA",
    n_samples: Annotated[int, typer.Option(help="Number of realisations.")] = 1,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    mode: Annotated[str, typer.Option(help="Output mode - column (modify in-place) or copy (new files).")] = "column",
    out_dir: Annotated[
        Optional[Directory], typer.Option(parser=Path, help="Output directory (only for copy mode).")
    ] = None,
):
    """
    Generates jackknife noise realisations from an MS.
    """
    # Lazy import to keep CLI fast and isolate logic
    from jackknify.core.realise import realise as realise_core

    realise_core(
        ms_file=str(ms_file),
        col=col,
        n_samples=n_samples,
        seed=seed,
        mode=mode,
        out_dir=str(out_dir) if out_dir else None,
    )
