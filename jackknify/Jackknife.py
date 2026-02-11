import jax
import jax.numpy as jnp

def jax_apply_flips(data : jnp.array, seed : int = 1995):
    """
    Applies the Jackknife sign flipping logic using JAX.
    Logic: Half of the data points are flipped (-1), half are kept (+1).
    Shuffling is random based on the seed.
    
    Args:
        data: Jax Numpy array of shape (N_row, N_chan, N_corr)
        seed: Integer seed for PRNG
    
    Returns:
        Jax numpy array with flips applied.
    """
    n_row, n_chan, _ = data.shape
    n_elements = n_row * n_chan
    
    key = jax.random.PRNGKey(seed)
    
    # Create flip mask (1D)
    # Half 1, Half -1
    ones = jnp.ones(n_elements)
    flips = ones.at[n_elements // 2:].set(-1.0)
    
    # Shuffle
    flips = jax.random.permutation(key, flips)
    
    # Reshape to match data (N_row, N_chan) and broadcast to N_corr
    flips_reshaped = flips.reshape((n_row, n_chan, 1))
    
    # Apply to data
    d_jax = data
    d_out = d_jax * flips_reshaped
    
    return d_out