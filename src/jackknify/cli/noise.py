from pathlib import Path
from typing import Annotated, NewType
import typer
from hip_cargo.utils.decorators import stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)

@stimela_cab(
    name="noise",
    info="Calculates a 'noise' cube (std dev) from a folder of FITS files.",
)
@stimela_output(
    name="out",
    dtype="File",
    info="The resulting noise cube FITS file.",
)
def noise(
    folder_path: Annotated[
        Directory, 
        typer.Argument(..., help="Folder containing input FITS files.")
    ],
    out: Annotated[
        File, 
        typer.Option(help="Output filename.")
    ] = Path("noise_cube.fits"),
):
    """
    Calculates a 'noise' cube (std dev) from a folder of FITS files.
    """
    from jackknify.core.CalcNoise import compute_noise_cube
    try:
        compute_noise_cube(str(folder_path), str(out))
        print(f"Noise cube written to {out}")
    except Exception as e:
        print(f"Error: {e}")