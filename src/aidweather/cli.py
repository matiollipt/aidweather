# SPDX-License-Identifier: Apache-2.0
"""
Command Line Interface for the AidWeather Power Client.

Exposes the following subcommand groups via the ``app`` Typer application:

- ``fetch``           — point data for a single lat/lon
- ``fetch-multi``     — parallel point data from a CSV file
- ``fetch-transect``  — data sampled along a geographic transect
- ``fetch-regional``  — bounding-box data on a 0.5° grid
- ``params``          — browse and describe NASA POWER parameters
- ``cache``           — inspect and clear the local SQLite cache
"""
from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from aidweather import __version__
from aidweather.client import AmbiguousDateError, PowerClient, parse_date_strict
from aidweather.config import cfg
from aidweather.geo import GeoCoordinate

app = typer.Typer(
    help="AidWeather CLI for fetching and analyzing agroclimatic data from NASA POWER.",
    no_args_is_help=True,
)
params_app = typer.Typer(help="Manage NASA POWER parameter catalogues.")
cache_app = typer.Typer(help="Manage the local SQLite cache.")

app.add_typer(params_app, name="params")
app.add_typer(cache_app, name="cache")

console = Console()

_SUPPORTED_OUTPUT_FORMATS = {"csv", "json", "parquet"}
_EXTENSION_TO_FORMAT = {
    ".csv": "csv",
    ".json": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def _print_preview(df: pd.DataFrame, n: int = 5) -> None:
    """Render the first *n* rows of *df* as a Rich table.

    Uses only ``rich`` (already a project dependency), so ``tabulate`` is
    not required.

    Args:
        df: The DataFrame to preview.
        n: Number of rows to display.
    """
    preview = df.head(n).reset_index()
    table = Table(show_header=True, header_style="bold cyan", show_lines=False)
    for col in preview.columns:
        table.add_column(str(col), no_wrap=True)
    for row_vals in preview.astype(str).values.tolist():
        table.add_row(*[str(v) for v in row_vals])
    console.print(table)


def _print_failed_points(failed: list[tuple[Any, str]], limit: int = 5) -> None:
    """Print the first *limit* failed points from *failed* with their error messages."""
    console.print(f"[yellow]Warning: {len(failed)} point(s) failed to fetch:[/yellow]")
    for point, error in failed[:limit]:
        console.print(f"  [yellow]•[/yellow] {point}: {error}")
    if len(failed) > limit:
        console.print(f"  ... and {len(failed) - limit} more.")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"aidweather {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the aidweather version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose console logging.",
        ),
    ] = False,
) -> None:
    """CLI entry point."""
    if verbose:
        package_logger = logging.getLogger("aidweather")
        if not any(type(h) is logging.StreamHandler for h in package_logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            package_logger.addHandler(stream_handler)
            package_logger.setLevel(logging.INFO)


def _parse_date(date_str: str) -> str:
    """Parse date and return in YYYYMMDD format."""
    try:
        return parse_date_strict(date_str).strftime("%Y%m%d")
    except AmbiguousDateError as e:
        raise typer.BadParameter(str(e)) from e
    except Exception as e:
        raise typer.BadParameter(f"Invalid date '{date_str}': {e}") from e


def _validate_resolution(resolution: str) -> None:
    """Print a styled error and exit if *resolution* is not ``'daily'`` or ``'hourly'``."""
    if resolution not in ("daily", "hourly"):
        console.print(
            "[bold red]❌ Error:[/bold red] resolution must be 'daily' or 'hourly'."
        )
        raise typer.Exit(code=1)


def _resolve_output_format(output: Path | None, fmt: str | None) -> str:
    """Resolve output format from file extension first, then --format."""
    fmt_normalized = fmt.lower() if fmt else None
    if fmt_normalized is not None and fmt_normalized not in _SUPPORTED_OUTPUT_FORMATS:
        raise typer.BadParameter(
            f"Unsupported format '{fmt}'. Expected one of: csv, json, parquet."
        )

    if output is not None:
        extension_format = _EXTENSION_TO_FORMAT.get(output.suffix.lower())
        if extension_format is not None:
            if fmt_normalized is not None and fmt_normalized != extension_format:
                console.print(
                    "[yellow]Warning:[/yellow] Output extension "
                    f"'{output.suffix}' implies {extension_format}; ignoring "
                    f"--format {fmt_normalized}."
                )
            return extension_format

    return fmt_normalized or "csv"


def _save_output(df: pd.DataFrame, output: Path | None, fmt: str | None) -> None:
    """Write DataFrame to the specified format."""
    if output is None:
        return

    resolved_fmt = _resolve_output_format(output, fmt)

    # ensure parent dir exists
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        if resolved_fmt == "csv":
            df.to_csv(output, index=df.index.name is not None)
            console.print(
                f"[bold green]✅ Success:[/bold green] Data saved to {output.resolve()}"
            )
        elif resolved_fmt == "json":
            # For JSON, we typically want records orientation. If date is index, reset it.
            if df.index.name:
                df_to_save = df.reset_index()
            else:
                df_to_save = df
            df_to_save.to_json(output, orient="records", date_format="iso")
            console.print(
                f"[bold green]✅ Success:[/bold green] Data saved to {output.resolve()}"
            )
        elif resolved_fmt == "parquet":
            df.to_parquet(output)
            console.print(
                f"[bold green]✅ Success:[/bold green] Data saved to {output.resolve()}"
            )
        else:
            console.print(
                f"[bold red]❌ Error:[/bold red] Unsupported format '{resolved_fmt}'."
            )
    except Exception as exc:
        console.print(f"[bold red]❌ Error saving to file:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def fetch(  # noqa: PLR0913
    lat: Annotated[
        float, typer.Option("--lat", help="Latitude of the point of interest.")
    ],
    lon: Annotated[
        float, typer.Option("--lon", help="Longitude of the point of interest.")
    ],
    start: Annotated[
        str, typer.Option("--start", help="Start date (e.g., YYYY-MM-DD).")
    ],
    end: Annotated[str, typer.Option("--end", help="End date (e.g., YYYY-MM-DD).")],
    params: Annotated[
        str,
        typer.Option(
            "--params", help="Comma-separated parameters (e.g., T2M,PRECTOTCORR)."
        ),
    ] = "T2M,PRECTOTCORR",
    resolution: Annotated[
        str,
        typer.Option("--resolution", help="Temporal resolution: 'daily' or 'hourly'."),
    ] = "daily",
    elevation: Annotated[
        float | None,
        typer.Option("--elevation", help="Optional site elevation in meters."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional path to save data.")
    ] = None,
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            help=(
                "Output format when --output has no recognized extension: "
                "'csv', 'json', or 'parquet'."
            ),
        ),
    ] = None,
    no_preview: Annotated[
        bool, typer.Option("--no-preview", help="Suppress data preview table.")
    ] = False,
    summarize: Annotated[
        bool, typer.Option("--summarize", help="Print summary panel.")
    ] = False,
):
    """Fetch weather data for a single geographic point."""
    param_list = [p.strip() for p in params.split(",") if p.strip()]
    parsed_start = _parse_date(start)
    parsed_end = _parse_date(end)

    _validate_resolution(resolution)

    try:
        client = PowerClient(temporal_api=cast(Literal["daily", "hourly"], resolution))
        df = client.get_point_data(
            lat=lat,
            lon=lon,
            start=parsed_start,
            end=parsed_end,
            params=param_list,
            elevation=elevation,
        )

        # Inject metadata for consistency with fetch-multi/transect
        if not df.empty:
            df = df.reset_index()
            df["lat"] = lat
            df["lon"] = lon
            if elevation is not None:
                df["elevation"] = elevation
            df = df.set_index("date")

    except Exception as e:
        console.print(f"[bold red]Error fetching data:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    if df.empty:
        console.print(
            "[yellow]Warning: The API returned no data for this request.[/yellow]"
        )
        return

    if not no_preview:
        console.print("\n[bold blue]--- Data Preview (First 5 Rows) ---[/bold blue]")
        _print_preview(df)

    if summarize:
        client.summarize(df)

    _save_output(df, output, fmt)


@app.command(name="fetch-multi")
def fetch_multi(  # noqa: PLR0913
    points_file: Annotated[
        Path,
        typer.Option(
            "--points-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="CSV file with 'lat' and 'lon' columns.",
        ),
    ],
    start: Annotated[
        str, typer.Option("--start", help="Start date (e.g., YYYY-MM-DD).")
    ],
    end: Annotated[str, typer.Option("--end", help="End date (e.g., YYYY-MM-DD).")],
    params: Annotated[
        str,
        typer.Option(
            "--params", help="Comma-separated parameters (e.g., T2M,PRECTOTCORR)."
        ),
    ] = "T2M,PRECTOTCORR",
    resolution: Annotated[
        str,
        typer.Option("--resolution", help="Temporal resolution: 'daily' or 'hourly'."),
    ] = "daily",
    workers: Annotated[
        int,
        typer.Option("--workers", help="Max concurrent requests (NASA recommends ≤5)."),
    ] = 5,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional path to save data.")
    ] = None,
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            help=(
                "Output format when --output has no recognized extension: "
                "'csv', 'json', or 'parquet'."
            ),
        ),
    ] = None,
    no_preview: Annotated[
        bool, typer.Option("--no-preview", help="Suppress data preview table.")
    ] = False,
    summarize: Annotated[
        bool, typer.Option("--summarize", help="Print summary panel.")
    ] = False,
):
    """Fetch weather data for multiple geographic points from a CSV file."""
    param_list = [p.strip() for p in params.split(",") if p.strip()]
    parsed_start = _parse_date(start)
    parsed_end = _parse_date(end)

    _validate_resolution(resolution)

    try:
        points_df = pd.read_csv(points_file)
        if "lat" not in points_df.columns or "lon" not in points_df.columns:
            console.print(
                "[bold red]❌ Error:[/bold red] CSV must contain 'lat' and 'lon' columns."
            )
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[bold red]❌ Error reading points file:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    try:
        client = PowerClient(temporal_api=cast(Literal["daily", "hourly"], resolution))
        df, failed = client.get_multi_point_data(
            points=points_df,
            start=parsed_start,
            end=parsed_end,
            params=param_list,
            max_workers=workers,
        )
    except Exception as e:
        console.print(f"[bold red]Error fetching multi-point data:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    if failed:
        _print_failed_points(failed)

    if df.empty:
        console.print(
            "[yellow]Warning: The API returned no data for these requests.[/yellow]"
        )
        return

    if not no_preview:
        console.print("\n[bold blue]--- Data Preview (First 5 Rows) ---[/bold blue]")
        _print_preview(df)

    if summarize:
        client.summarize(df)

    _save_output(df, output, fmt)


@app.command(name="fetch-transect")
def fetch_transect(  # noqa: PLR0913
    lat_start: Annotated[
        float, typer.Option("--lat-start", help="Latitude of the transect start point.")
    ],
    lon_start: Annotated[
        float,
        typer.Option("--lon-start", help="Longitude of the transect start point."),
    ],
    lat_end: Annotated[
        float, typer.Option("--lat-end", help="Latitude of the transect end point.")
    ],
    lon_end: Annotated[
        float, typer.Option("--lon-end", help="Longitude of the transect end point.")
    ],
    start: Annotated[
        str, typer.Option("--start", help="Start date (e.g., YYYY-MM-DD).")
    ],
    end: Annotated[str, typer.Option("--end", help="End date (e.g., YYYY-MM-DD).")],
    params: Annotated[
        str,
        typer.Option(
            "--params", help="Comma-separated parameters (e.g., T2M,PRECTOTCORR)."
        ),
    ] = "T2M,PRECTOTCORR",
    resolution: Annotated[
        str,
        typer.Option("--resolution", help="Temporal resolution: 'daily' or 'hourly'."),
    ] = "daily",
    num_points: Annotated[
        int | None,
        typer.Option(
            "--num-points",
            help=(
                "Number of sampling points along the transect. "
                "Takes priority over --spacing-km when both are given."
            ),
        ),
    ] = None,
    spacing_km: Annotated[
        float | None,
        typer.Option(
            "--spacing-km",
            help=(
                "Approximate spacing between sample points in km. "
                "Used to derive num-points when --num-points is omitted. "
                "Minimum effective spacing is ~55 km (NASA POWER 0.5° grid)."
            ),
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Max concurrent requests (NASA recommends ≤5)."),
    ] = 5,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional path to save data.")
    ] = None,
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            help=(
                "Output format when --output has no recognized extension: "
                "'csv', 'json', or 'parquet'."
            ),
        ),
    ] = None,
    no_preview: Annotated[
        bool, typer.Option("--no-preview", help="Suppress data preview table.")
    ] = False,
    summarize: Annotated[
        bool, typer.Option("--summarize", help="Print summary panel.")
    ] = False,
):
    """Fetch weather data along a 1D transect between two geographic endpoints.

    Points are sampled evenly along the straight-line path from the start to
    the end coordinate. Provide either --num-points or --spacing-km to control
    sampling density. The minimum effective spacing is ~55 km (NASA POWER
    native 0.5° grid resolution); finer requests are clamped automatically.
    """
    param_list = [p.strip() for p in params.split(",") if p.strip()]
    parsed_start = _parse_date(start)
    parsed_end = _parse_date(end)

    _validate_resolution(resolution)

    if num_points is None and spacing_km is None:
        console.print(
            "[bold red]❌ Error:[/bold red] Provide --num-points or --spacing-km."
        )
        raise typer.Exit(code=1)

    try:
        coord_a = GeoCoordinate.from_decimal(lat_start, lon_start)
        coord_b = GeoCoordinate.from_decimal(lat_end, lon_end)
        client = PowerClient(temporal_api=cast(Literal["daily", "hourly"], resolution))
        df = client.get_transect_data_from_coordinates(
            coord_a=coord_a,
            coord_b=coord_b,
            start=parsed_start,
            end=parsed_end,
            params=param_list,
            num_points=num_points,
            spacing_km=spacing_km,
            max_workers=workers,
        )
    except Exception as e:
        console.print(f"[bold red]Error fetching transect data:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    if df.empty:
        console.print(
            "[yellow]Warning: The API returned no data for this request.[/yellow]"
        )
        return

    if not no_preview:
        console.print("\n[bold blue]--- Data Preview (First 5 Rows) ---[/bold blue]")
        _print_preview(df)

    if summarize:
        client.summarize(df)

    _save_output(df, output, fmt)


@app.command(name="fetch-regional")
def fetch_regional(  # noqa: PLR0913
    lat_min: Annotated[
        float,
        typer.Option("--lat-min", help="Southern edge of the bounding box (latitude)."),
    ],
    lat_max: Annotated[
        float,
        typer.Option("--lat-max", help="Northern edge of the bounding box (latitude)."),
    ],
    lon_min: Annotated[
        float,
        typer.Option("--lon-min", help="Western edge of the bounding box (longitude)."),
    ],
    lon_max: Annotated[
        float,
        typer.Option("--lon-max", help="Eastern edge of the bounding box (longitude)."),
    ],
    start: Annotated[
        str, typer.Option("--start", help="Start date (e.g., YYYY-MM-DD).")
    ],
    end: Annotated[str, typer.Option("--end", help="End date (e.g., YYYY-MM-DD).")],
    params: Annotated[
        str,
        typer.Option(
            "--params",
            help="Single parameter for regional requests (e.g., T2M).",
        ),
    ] = "T2M",
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional path to save data.")
    ] = None,
    fmt: Annotated[
        str | None,
        typer.Option(
            "--format",
            help=(
                "Output format when --output has no recognized extension: "
                "'csv', 'json', or 'parquet'."
            ),
        ),
    ] = None,
    no_preview: Annotated[
        bool, typer.Option("--no-preview", help="Suppress data preview table.")
    ] = False,
    summarize: Annotated[
        bool, typer.Option("--summarize", help="Print summary panel.")
    ] = False,
):
    """Fetch weather data for a regional bounding box (0.5° grid).

    The NASA POWER regional API returns data on a 0.5° × 0.5° grid within
    the specified bounding box. The box must not exceed 4.5° on either axis,
    and only one parameter can be requested per call.
    """
    param_list = [p.strip() for p in params.split(",") if p.strip()]
    parsed_start = _parse_date(start)
    parsed_end = _parse_date(end)

    try:
        # The NASA POWER regional API only supports daily temporal resolution.
        client = PowerClient(temporal_api="daily")
        df = client.get_regional_data(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            start=parsed_start,
            end=parsed_end,
            params=param_list,
        )
    except ValueError as e:
        console.print(f"[bold red]❌ Validation Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]Error fetching regional data:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    if df.empty:
        console.print(
            "[yellow]Warning: The API returned no data for this request.[/yellow]"
        )
        return

    if not no_preview:
        console.print("\n[bold blue]--- Data Preview (First 5 Rows) ---[/bold blue]")
        _print_preview(df)

    if summarize:
        console.print(
            "[yellow]Note:[/yellow] --summarize is not supported for regional data "
            "(uses a multi-column structure, not a DatetimeIndex). Skipping."
        )

    _save_output(df, output, fmt)


@params_app.command("list")
def params_list(
    group: Annotated[
        str,
        typer.Option(
            "--group", help="Parameter group to list (e.g., 'default', 'all')."
        ),
    ] = "default",
):
    """List available NASA POWER parameters."""
    available_groups = cfg.param_groups()
    if group not in available_groups:
        avail_str = ", ".join(available_groups)
        console.print(
            f"[bold red]❌ Error:[/bold red] Unknown group '{group}'. Available: {avail_str}"
        )
        raise typer.Exit(code=1)

    param_dict = cfg.params(group)

    table = Table(
        title=f"NASA POWER Parameters (Group: {group})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Code", style="bold")
    table.add_column("Short Name")

    for code, name in param_dict.items():
        table.add_row(code, name)

    console.print(table)


@params_app.command("describe")
def params_describe(
    code: Annotated[
        str, typer.Argument(help="Parameter code to describe (e.g., T2M).")
    ],
):
    """Print the full science description for a given parameter code."""
    descriptions = cfg.param_descriptions()

    # Allow case-insensitive lookups
    code_upper = code.upper()

    if code_upper not in descriptions:
        console.print(
            f"[bold red]❌ Error:[/bold red] Unknown parameter code '{code_upper}'. "
            "Try 'aidweather params list --group all'"
        )
        raise typer.Exit(code=1)

    console.print(f"\n[bold]{descriptions[code_upper]}[/bold]\n")


@cache_app.command("info")
def cache_info():
    """Show details about the local SQLite cache."""
    cache_cfg = cfg.cache_config()
    enabled = cache_cfg.get("enabled", False)
    cache_dir = cache_cfg.get("path", ".")
    db_path = Path(cache_dir) / "aidweather_cache.db"

    console.print(f"[bold]Cache Enabled:[/bold] {enabled}")
    console.print(f"[bold]Cache Path:[/bold] {db_path.resolve()}")
    env_override = os.environ.get("AIDWEATHER_CACHE_DIR")
    if env_override:
        console.print("[bold]Path Source:[/bold] AIDWEATHER_CACHE_DIR env var")
    else:
        console.print(
            "[bold]Path Override:[/bold] set AIDWEATHER_CACHE_DIR to use a custom location"
        )

    if not enabled:
        return

    if not db_path.exists():
        console.print("[bold]Cache Status:[/bold] Empty — no database found yet.")
        return

    size_bytes = db_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    console.print(f"[bold]File Size:[/bold] {size_mb:.2f} MB ({size_bytes:,} bytes)")

    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM cache")
            count = cur.fetchone()[0]
            console.print(f"[bold]Cached Entries:[/bold] {count}")

            if count > 0:
                cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM cache")
                oldest, newest = cur.fetchone()
                console.print(f"[bold]Oldest Entry:[/bold] {oldest}")
                console.print(f"[bold]Newest Entry:[/bold] {newest}")

                cur.execute("SELECT SUM(LENGTH(data)) FROM cache")
                raw_compressed = cur.fetchone()[0] or 0
                console.print(
                    f"[bold]Compressed Data:[/bold] {raw_compressed / (1024 * 1024):.2f} MB"
                    f" (gzip, stored in BLOB)"
                )
    except sqlite3.Error as e:
        console.print(f"[bold red]Error reading cache DB:[/bold red] {e}")


@cache_app.command("clear")
def cache_clear(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Confirm deletion without prompting.")
    ] = False,
):
    """Clear the local SQLite cache."""
    cache_cfg = cfg.cache_config()
    cache_dir = cache_cfg.get("path", ".")
    db_path = Path(cache_dir) / "aidweather_cache.db"

    if not db_path.exists():
        console.print("Cache is already empty.")
        return

    if not yes:
        confirm = typer.confirm("Are you sure you want to delete the cache database?")
        if not confirm:
            console.print("Operation cancelled.")
            raise typer.Exit()

    try:
        db_path.unlink()
        console.print("[bold green]✅ Cache cleared successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Error clearing cache:[/bold red] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
