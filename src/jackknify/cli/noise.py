from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="noise",
    info="Calculates a 'noise' cube (std dev) from a folder of FITS files.",
)
@stimela_output(
    dtype="File",
    name="out_cube",
    info="The resulting noise cube FITS file.",
)
def noise(
    folder_path: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Folder containing input FITS files.",
        ),
    ],
    out_cube: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="The resulting noise cube FITS file.",
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
    Calculates a 'noise' cube (std dev) from a folder of FITS files.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                noise,
                dict(
                    folder_path=folder_path,
                    out_cube=out_cube,
                ),
            )

            # Lazy import the core implementation
            from jackknify.core.noise import noise as noise_core  # noqa: E402

            # Call the core function with all parameters
            noise_core(
                folder_path,
                out_cube=out_cube,
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
        noise,
        dict(
            folder_path=folder_path,
            out_cube=out_cube,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
