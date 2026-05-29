# SPDX-License-Identifier: Apache-2.0
"""
Command Line Interface for the AidWeather Power Client.
"""

import sqlite3
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from aidweather.config import cfg
from rich.console import Console
from rich.table import Table

from aidweather.client import PowerClient

app = typer.Typer(
    help="CLI for fetching and analyzing agroclimatic data from NASA POWER.",
    no_args_is_help=True,
)
params_app = typer.Typer(help="Manage NASA POWER parameter catalogues.")
cache_app = typer.Typer(help="Manage the local SQLite cache.")

app.add_typer(params_app, name="params")
app.add_typer(cache_app, name="cache")

console = Console()


def _parse_date(date_str: str) -> str:
    """Parse date and return in YYYYMMDD format."""
    try:
        return pd.to_datetime(date_str).strftime("%Y%m%d")
    except Exception as e:
        raise typer.BadParameter(
            f"Invalid date: '{date_str}'. Expected format like YYYY-MM-DD."
        ) from e


def _save_output(df: pd.DataFrame, output: Path | None, fmt: str) -> None:
    """Write DataFrame to the specified format."""
    if output is None:
        return

    # ensure parent dir exists
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        if fmt == "csv":
            df.to_csv(output, index=df.index.name is not None)
            console.print(
                f"[bold green]✅ Success:[/bold green] Data saved to {output.resolve()}"
            )
        elif fmt == "json":
            # For JSON, we typically want records orientation. If date is index, reset it.
            if df.index.name:
                df_to_save = df.reset_index()
            else:
                df_to_save = df
            df_to_save.to_json(output, orient="records", date_format="iso")
            console.print(
                f"[bold green]✅ Success:[/bold green] Data saved to {output.resolve()}"
            )
        elif fmt == "parquet":
            df.to_parquet(output)
            console.print(
                f"[bold green]✅ Success:[/bold green] Data saved to {output.resolve()}"
            )
        else:
            console.print(f"[bold red]❌ Error:[/bold red] Unsupported format '{fmt}'.")
    except Exception as exc:
        console.print(f"[bold red]❌ Error saving to file:[/bold red] {exc}")
        raise typer.Exit(code=1)


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
        str, typer.Option("--format", help="Output format: 'csv', 'json', 'parquet'.")
    ] = "csv",
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

    if resolution not in ["daily", "hourly"]:
        console.print(
            "[bold red]❌ Error:[/bold red] resolution must be 'daily' or 'hourly'."
        )
        raise typer.Exit(code=1)

    try:
        client = PowerClient(temporal_api=resolution)
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
        raise typer.Exit(code=1)

    if df.empty:
        console.print(
            "[yellow]Warning: The API returned no data for this request.[/yellow]"
        )
        return

    if not no_preview:
        console.print("\n[bold blue]--- Data Preview (First 5 Rows) ---[/bold blue]")
        console.print(df.head().to_markdown())

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
        int, typer.Option("--workers", help="Max concurrent requests.")
    ] = 8,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional path to save data.")
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", help="Output format: 'csv', 'json', 'parquet'.")
    ] = "csv",
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

    if resolution not in ["daily", "hourly"]:
        console.print(
            "[bold red]❌ Error:[/bold red] resolution must be 'daily' or 'hourly'."
        )
        raise typer.Exit(code=1)

    try:
        points_df = pd.read_csv(points_file)
        if "lat" not in points_df.columns or "lon" not in points_df.columns:
            console.print(
                "[bold red]❌ Error:[/bold red] CSV must contain 'lat' and 'lon' columns."
            )
            raise typer.Exit(code=1)

        points: list[dict[str, Any]] = []
        for _, row in points_df.iterrows():
            pt = {"lat": float(row["lat"]), "lon": float(row["lon"])}
            if "elevation" in points_df.columns and not pd.isna(row["elevation"]):
                pt["elevation"] = float(row["elevation"])
            if "name" in points_df.columns and not pd.isna(row["name"]):
                pt["name"] = str(row["name"])
            points.append(pt)
    except Exception as e:
        console.print(f"[bold red]❌ Error reading points file:[/bold red] {e}")
        raise typer.Exit(code=1)

    try:
        client = PowerClient(temporal_api=resolution)
        df, _failed = client.get_multi_point_data(
            points=points,
            start=parsed_start,
            end=parsed_end,
            params=param_list,
            max_workers=workers,
        )
    except Exception as e:
        console.print(f"[bold red]Error fetching multi-point data:[/bold red] {e}")
        raise typer.Exit(code=1)

    if df.empty:
        console.print(
            "[yellow]Warning: The API returned no data for these requests.[/yellow]"
        )
        return

    if not no_preview:
        console.print("\n[bold blue]--- Data Preview (First 5 Rows) ---[/bold blue]")
        console.print(df.head().to_markdown())

    if summarize:
        client.summarize(df)

    _save_output(df, output, fmt)


@app.command(name="fetch-transect")
def fetch_transect(  # noqa: PLR0913
    lat: Annotated[float, typer.Option("--lat", help="Latitude of the center point.")],
    lon: Annotated[float, typer.Option("--lon", help="Longitude of the center point.")],
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
    axis: Annotated[
        str, typer.Option("--axis", help="Expansion axis: 'lat' or 'lon'.")
    ] = "lat",
    distance_km: Annotated[
        float, typer.Option("--distance-km", help="Total transect length in km.")
    ] = 10.0,
    num_points: Annotated[
        int, typer.Option("--num-points", help="Number of sampling points.")
    ] = 10,
    workers: Annotated[
        int, typer.Option("--workers", help="Max concurrent requests.")
    ] = 8,
    elevation: Annotated[
        float | None,
        typer.Option("--elevation", help="Optional uniform site elevation in meters."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional path to save data.")
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", help="Output format: 'csv', 'json', 'parquet'.")
    ] = "csv",
    no_preview: Annotated[
        bool, typer.Option("--no-preview", help="Suppress data preview table.")
    ] = False,
    summarize: Annotated[
        bool, typer.Option("--summarize", help="Print summary panel.")
    ] = False,
):
    """Fetch weather data across a linear transect of points expanding from a center coordinate."""
    param_list = [p.strip() for p in params.split(",") if p.strip()]
    parsed_start = _parse_date(start)
    parsed_end = _parse_date(end)

    if resolution not in ["daily", "hourly"]:
        console.print(
            "[bold red]❌ Error:[/bold red] resolution must be 'daily' or 'hourly'."
        )
        raise typer.Exit(code=1)

    if axis not in ["lat", "lon"]:
        console.print("[bold red]❌ Error:[/bold red] axis must be 'lat' or 'lon'.")
        raise typer.Exit(code=1)

    try:
        client = PowerClient(temporal_api=resolution)
        df = client.get_expanded_point_data(
            lat=lat,
            lon=lon,
            start=parsed_start,
            end=parsed_end,
            params=param_list,
            axis=axis,
            distance_km=distance_km,
            num_points=num_points,
            elevation=elevation,
            max_workers=workers,
        )
    except Exception as e:
        console.print(f"[bold red]Error fetching transect data:[/bold red] {e}")
        raise typer.Exit(code=1)

    if df.empty:
        console.print(
            "[yellow]Warning: The API returned no data for this request.[/yellow]"
        )
        return

    if not no_preview:
        console.print("\n[bold blue]--- Data Preview (First 5 Rows) ---[/bold blue]")
        console.print(df.head().to_markdown())

    if summarize:
        client.summarize(df)

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
    if not enabled:
        return

    if not db_path.exists():
        console.print(
            f"[bold]Cache Status:[/bold] Empty/Not found at {db_path.resolve()}"
        )
        return

    size_mb = db_path.stat().st_size / (1024 * 1024)
    console.print(f"[bold]Cache File:[/bold] {db_path.resolve()}")
    console.print(f"[bold]Size:[/bold] {size_mb:.2f} MB")

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cache")
        count = cur.fetchone()[0]
        console.print(f"[bold]Entries:[/bold] {count}")
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
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
