import jax

# Ensure JAX uses 64-bit precision globally
jax.config.update('jax_enable_x64', True)