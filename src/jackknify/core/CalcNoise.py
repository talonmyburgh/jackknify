import os
import numpy as np
from tqdm import tqdm
from .FitsHandler import FitsWrapper

def compute_noise_cube(folder_path, out_file):
    """
    Reads all .fits files in a folder and computes the standard deviation
    across the file axis. 
    
    Uses Numpy for efficient memory buffering (loading) and JAX for 
    high-performance calculation.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    num_files = len(files)
    
    if num_files == 0:
        raise FileNotFoundError(f"No .fits files found in {folder_path}")

    print(f"Found {num_files} files. Inspecting first file for shape...")
    
    # Use the first file to initialize the stack
    first_wrapper = FitsWrapper(os.path.join(folder_path, files[0]))
    header = first_wrapper.header
    data_shape = first_wrapper.data.shape
    dtype = first_wrapper.data.dtype
    
    # Pre-allocate the stack using standard Numpy (Mutable)
    print(f"Pre-allocating stack with shape {(num_files, *data_shape)}...")
    stack = np.zeros((num_files, *data_shape), dtype=dtype)
    
    # Fill the stack
    stack[0] = first_wrapper.data
    
    for i, f in tqdm(enumerate(files[1:], start=1), total=num_files-1, desc="Loading Fits"):
        wrapper = FitsWrapper(os.path.join(folder_path, f))
        stack[i] = wrapper.data
    
    print("Computing standard deviation with JAX...")
    
    # Calculate Standard Deviation using JAX
    std_cube = np.nanstd(stack, axis=0)
    
    print(f"Saving to {out_file}...")
    FitsWrapper.write_cube(out_file, std_cube, header)