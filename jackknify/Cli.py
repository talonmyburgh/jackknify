import os
import click
from tqdm import tqdm
from .MSHandler import MSWrapper
from .Jackknife import jax_apply_flips
from .CalcNoise import compute_noise_cube

@click.group()
def cli():
    """JAXknife: A simple tool for visibility noise realizations."""
    pass

@cli.command()
@click.argument('ms_file', type=click.Path(exists=True))
@click.option('--col', default='DATA', help='Input data column name.')
@click.option('--n-samples', '-n', default=1, help='Number of realizations.')
@click.option('--seed', default=42, help='Random seed.')
@click.option('--mode', type=click.Choice(['column', 'copy']), default='column', 
              help='Output mode: "column" writes DATA_JACK_i cols to same MS, "copy" creates new MS files.')
@click.option('--out-dir', default='./jackknife_output', help='Output directory (only for copy mode).')
def realize(ms_file, col, n_samples, seed, mode, out_dir):
    """
    Generates jackknife noise realizations from an MS.
    """
    wrapper = MSWrapper(ms_file)
    print(f"Reading {col} from {ms_file}...")
    original_data = wrapper.get_data(col)
    
    if mode == 'copy' and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in tqdm(range(n_samples), desc="Generating Realizations"):
        current_seed = seed + i
        jacked_data = jax_apply_flips(original_data, current_seed)
        
        if mode == 'column':
            out_col_name = f"{col}_JACK_{i}"
            wrapper.write_column(out_col_name, jacked_data, desc_template_col=col)
        
        elif mode == 'copy':
            ms_name = os.path.basename(ms_file).replace('.ms', '')
            new_ms_path = os.path.join(out_dir, f"{ms_name}_JACK_{i}.ms")
            
            new_wrapper = wrapper.create_copy(new_ms_path)
            new_wrapper.write_column('DATA', jacked_data, desc_template_col=col)

    print("Done.")

@cli.command()
@click.argument('folder_path', type=click.Path(exists=True))
@click.option('--out', default='noise_cube.fits', help='Output filename.')
def noise(folder_path, out):
    """
    Calculates a 'noise' cube (std dev) from a folder of FITS files.
    """
    try:
        compute_noise_cube(folder_path, out)
        print(f"Noise cube written to {out}")
    except Exception as e:
        print(f"Error: {e}")