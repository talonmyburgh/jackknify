from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, stimela_cab, stimela_output

MS = NewType("MS", Path)


@stimela_cab(
    name="make_ms",
    info="Creates a simple mock MS filled with 1s for testing.",
)
@stimela_output(
    dtype="MS",
    name="out_ms",
    info="The resulting mock Measurement Set.",
    implicit="{ms_file}",
)
def make_ms(
    ms_file: Annotated[
        str,
        typer.Option(
            ...,
            help="Path to create the mock MS.",
        ),
    ],
    rows: Annotated[
        int,
        typer.Option(
            help="",
        ),
    ] = 100,
    chans: Annotated[
        int,
        typer.Option(
            help="",
        ),
    ] = 16,
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
    Creates a simple mock MS filled with 1s for testing.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                make_ms,
                dict(
                    ms_file=ms_file,
                    rows=rows,
                    chans=chans,
                ),
            )

            # Lazy import the core implementation
            from jackknify.core.make_ms import make_ms as make_ms_core  # noqa: E402

            # Call the core function with all parameters
            make_ms_core(
                ms_file,
                rows=rows,
                chans=chans,
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
        make_ms,
        dict(
            ms_file=ms_file,
            rows=rows,
            chans=chans,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
