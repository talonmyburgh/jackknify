Jack-knife
==========
[![DOI](https://zenodo.org/badge/593247898.svg)](https://zenodo.org/doi/10.5281/zenodo.12516584)
[![Docs](https://img.shields.io/badge/docs-v1.0.0-2ea44f)](https://joshiwavm.github.io/jackknify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


``jackknify``is a Python-based package that jackknifes Measurement Set visibilities to create noise realizations from the observations.

Methodology
==========

Jackknifing is a simple but effective tool to characterize the underlying noise distribution of any type of data set. This tool specifically is implemented for interferometric data. ``jackknify`` splits half the visibilities randomly in two subsets, then multiplies one half with -1 so that when the data is binned, any signal present in the data is averaged out. This creates observation-specific noise realization of the data, which can be used to for instance, sample the likelihood a false detection.

The full methodology can be found [here](https://ui.adsabs.harvard.edu/abs/2025A%26A...695A.204V/abstract) and is also used in [this work](https://arxiv.org/abs/2210.03754).

Installation
============

``jackknify`` ships in two install modes.

**Lightweight** (default) pulls only ``hip-cargo`` and ``typer``:

    pip install jackknify

Every command still works in this mode — the CLI wrappers dispatch into the
project's container image (``ghcr.io/talonmyburgh/jackknify``) when the heavy
dependencies are not importable. This is the right mode for Stimela and for
machines that only need to launch commands.

**Full** adds the runtime dependencies needed to execute natively:

    pip install jackknify[full]

or from source:

    git clone https://github.com/talonmyburgh/jackknify
    cd jackknify
    uv sync --all-extras

Use ``--backend native`` to force in-process execution and surface an
``ImportError`` rather than falling back to a container, or ``--backend
docker`` / ``apptainer`` / ``podman`` / ``singularity`` to skip the native
attempt entirely.

Command-line interface
======================

Commands are defined once and exposed both as a CLI and as Stimela cabs
(generated into ``src/jackknify/cabs/``). Required parameters are passed as
options rather than positionally:

    jackknify realise --ms-file /path/to/observation.ms --n-samples 5
    jackknify noise --folder-path noise_images --out noise_cube.fits
    jackknify make-ms --ms-file mock.ms --rows 100 --chans 16

Path parameters are parsed with hip-cargo's ``parse_upath``, so remote URIs
(``s3://``, ``gs://``, ``az://``) are accepted at the CLI boundary. Note that
the ``core`` implementations currently read and write through local filesystem
calls (``os.listdir``, ``casacore.tables.table``), so **remote paths are not
yet supported end to end** — pass local paths for now.

**Note:** prior to the hip-cargo 0.3.0 conversion, ``ms-file`` and
``folder-path`` were positional arguments and ``make-ms`` was called
``make-test-ms``.

## Dependencies

``jackknify`` uses ``casatask`` and ``casatools`` to interface with CASA measurements. ``casatask`` and ``casatools`` requires ``casadata`` to load. Sadly, this is a  ~350 MB sized file making the installment a bit slow. Further, when performing line searches, we make use of the package ``interferopy``, which is a Python-based package for common tasks used in the observational radio/mm interferometry data analysis.

## Trouble shooting casatask installation (if needed)

If you want to run `jackknify` on a Mac with an Apple Silicon chip, run it in a Rosetta terminal. To open a Rosetta session in your terminal, run:

    /usr/bin/arch -x86_64 /bin/zsh --login

Further, `casadata` will download and store examples sets into the folder:  ~/.casa/data. However, it might not have permission from the local machine to do so. If such an error comes up. Just run:

    mkdir ~/.casa/data

To make the folder. That should solve most problems.


Documentation
============

For your convenience, there are notebooks on how to run and use ``jackknify`` for line inference. You can find them in the docs/notebooks folder. Also, check out the documentation [here](https://joshiwavm.github.io/jackknify/).
