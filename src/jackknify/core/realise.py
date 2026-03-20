import os
from typing import Optional

from tqdm import tqdm

from jackknify.core.jackknife import jax_apply_flips
from jackknify.core.ms_handler import MSWrapper


def realise(ms_file: str, col: str, n_samples: int, seed: int, mode: str, out_dir: Optional[str] = None):
    """Generates jackknife noise realisations from an MS."""
    wrapper = MSWrapper(ms_file)
    print(f"Reading {col} from {ms_file}...")
    original_data = wrapper.get_data(col)

    if mode == "copy" and out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    for i in tqdm(range(n_samples), desc="Generating realisations"):
        current_seed = seed + i
        jacked_data = jax_apply_flips(original_data, current_seed)

        if mode == "column":
            out_col_name = f"{col}_JACK_{i}"
            wrapper.write_column(out_col_name, jacked_data, desc_template_col=col)

        elif mode == "copy" and out_dir:
            ms_name = os.path.basename(ms_file).replace(".ms", "")
            new_ms_path = os.path.join(out_dir, f"{ms_name}_JACK_{i}.ms")
            new_wrapper = wrapper.create_copy(new_ms_path)
            new_wrapper.write_column("DATA", jacked_data, desc_template_col=col)

    print("Done.")
