import os

from tqdm import tqdm

from jackknify.core.jackknife import jax_apply_flips
from jackknify.core.ms_handler import MSWrapper


def realise(
    ms_file: str | os.PathLike,
    col: str,
    n_samples: int,
    seed: int,
    out_dir: str | os.PathLike | None = None,
):
    """Generates jackknife noise realisations from an MS."""
    ms_file = str(ms_file)
    out_dir = str(out_dir) if out_dir is not None else None
    wrapper = MSWrapper(ms_file)
    print(f"Reading {col} from {ms_file}...")
    original_data = wrapper.get_data(col)

    for i in tqdm(range(n_samples), desc="Generating realisations"):
        current_seed = seed + i
        jacked_data = jax_apply_flips(original_data, current_seed)

        out_col_name = f"{col}_JACK_{i}"
        wrapper.write_column(out_col_name, jacked_data, desc_template_col=col)

    print("Done.")
