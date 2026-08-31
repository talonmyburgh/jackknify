import os

from jackknify.core.calcnoise import compute_noise_cube


def noise(folder_path: str | os.PathLike, out: str | os.PathLike):
    """Calculates a 'noise' cube (std dev) from a folder of FITS files."""
    folder_path = str(folder_path)
    out = str(out)
    try:
        compute_noise_cube(folder_path, out)
        print(f"Noise cube written to {out}")
    except Exception as e:
        print(f"Error: {e}")
