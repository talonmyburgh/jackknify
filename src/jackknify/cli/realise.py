from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
MS = NewType("MS", Path)


@stimela_cab(
    name="realise",
    info="Generates jackknife noise realisations from a Measurement Set.",
)
@stimela_output(
    dtype="Directory",
    name="out_dir",
    info="Output directory (only used if mode is 'copy').",
)
def realise(
    ms_file: Annotated[
        MS,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Input Measurement Set.",
        ),
        StimelaMeta(
            writable=True,
        ),
    ],
    col: Annotated[
        str,
        typer.Option(
            help="Input data column name.",
        ),
    ] = "DATA",
    n_samples: Annotated[
        int,
        typer.Option(
            help="Number of realisations.",
        ),
    ] = 1,
    seed: Annotated[
        int,
        typer.Option(
            help="Random seed.",
        ),
    ] = 42,
    mode: Annotated[
        str,
        typer.Option(
            help="Output mode - column (modify in-place) or copy (new files).",
        ),
    ] = "column",
    out_dir: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Output directory (only for copy mode).",
        ),
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Generates jackknife noise realisations from a Measurement Set.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                realise,
                dict(
                    ms_file=ms_file,
                    col=col,
                    n_samples=n_samples,
                    seed=seed,
                    mode=mode,
                    out_dir=out_dir,
                ),
            )

            # Lazy import the core implementation
            from jackknify.core.realise import realise as realise_core  # noqa: E402

            # Call the core function with all parameters
            realise_core(
                ms_file,
                col=col,
                n_samples=n_samples,
                seed=seed,
                mode=mode,
                out_dir=out_dir,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("jackknify")
    if image is None:
        raise RuntimeError("No Container URL in jackknify metadata.")

    run_in_container(
        realise,
        dict(
            ms_file=ms_file,
            col=col,
            n_samples=n_samples,
            seed=seed,
            mode=mode,
            out_dir=out_dir,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
