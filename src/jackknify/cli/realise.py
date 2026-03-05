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
    ms_file: Annotated[
        MS, 
        typer.Argument(..., help="Input Measurement Set.")
    ],
    col: Annotated[
        str, 
        typer.Option(help="Input data column name.")
    ] = "DATA",
    n_samples: Annotated[
        int, 
        typer.Option(help="Number of realisations.")
    ] = 1,
    seed: Annotated[
        int, 
        typer.Option(help="Random seed.")
    ] = 42,
    mode: Annotated[
        str, 
        typer.Option(help="Output mode - column (modify in-place) or copy (new files).")
    ] = "column",
    out_dir: Annotated[
        Optional[Directory], 
        typer.Option(help="Output directory (only for copy mode).")
    ] = None,
):
    """
    Generates jackknife noise realisations from an MS.
    """
    # Lazy import to keep CLI fast
    from jackknify.core.MSHandler import MSWrapper
    from jackknify.core.Jackknife import jax_apply_flips
    from tqdm import tqdm
    import os

    wrapper = MSWrapper(str(ms_file))
    print(f"Reading {col} from {ms_file}...")
    original_data = wrapper.get_data(col)
    
    if mode == 'copy' and out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    for i in tqdm(range(n_samples), desc="Generating realisations"):
        current_seed = seed + i
        jacked_data = jax_apply_flips(original_data, current_seed)
        
        if mode == 'column':
            out_col_name = f"{col}_JACK_{i}"
            wrapper.write_column(out_col_name, jacked_data, desc_template_col=col)
        
        elif mode == 'copy' and out_dir:
            ms_name = os.path.basename(ms_file).replace('.ms', '')
            new_ms_path = os.path.join(out_dir, f"{ms_name}_JACK_{i}.ms")
            new_wrapper = wrapper.create_copy(new_ms_path)
            new_wrapper.write_column('DATA', jacked_data, desc_template_col=col)

    print("Done.")