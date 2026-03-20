from jackknify.core.CalcNoise import compute_noise_cube


def noise(folder_path: str, out: str):
    """Calculates a 'noise' cube (std dev) from a folder of FITS files."""
    try:
        compute_noise_cube(folder_path, out)
        print(f"Noise cube written to {out}")
    except Exception as e:
        print(f"Error: {e}")
