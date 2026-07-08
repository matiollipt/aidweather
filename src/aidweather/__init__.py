# SPDX-License-Identifier: Apache-2.0
"""
aidweather
~~~~~~~~~~

Weather data retrieval and validation for agricultural and environmental applications.

Provides a beta-ready client for NASA's POWER API, with a local SQLite cache,
retry logic, parallel fetching, geospatial coordinate utilities, and a CLI.

Quick start:
    >>> from aidweather import PowerClient
    >>> client = PowerClient(temporal_api="daily")
    >>> df = client.get_point_data(
    ...     lat=-23.55, lon=-46.63,
    ...     start="2023-01-01", end="2023-12-31",
    ...     params=["T2M", "PRECTOTCORR"],
    ... )
    >>> print(df.head())

License:
    Distributed under the Apache-2.0 license.
"""

__version__ = "0.1.2"
__author__ = "Cleverson Matiolli"
__url__ = "https://github.com/matiollipt/aidweather"

import logging
import os

from aidweather.client import PowerClient
from aidweather.config import cfg, get_config
from aidweather.geo import GeoCoordinate, normalize_coord_input
from aidweather.utils import ensure_date_column

__all__ = [
    "PowerClient",
    "GeoCoordinate",
    "normalize_coord_input",
    "cfg",
    "get_config",
    "ensure_date_column",
]

_logger = logging.getLogger("aidweather")

# Prevent "No handler found" warnings if the library user has not configured logging.
_logger.addHandler(logging.NullHandler())


def _configure_file_logging() -> None:
    """Attach a FileHandler to the package logger per logging_config in config.json.

    A no-op unless ``logging_config.enabled`` is true (off by default), so
    importing the package never writes to disk without explicit opt-in.
    """
    log_settings = cfg.logging_config()
    if not log_settings.get("enabled"):
        return

    log_path = log_settings["filename"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    level = getattr(logging, str(log_settings.get("level", "INFO")).upper(), logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    _logger.addHandler(handler)
    _logger.setLevel(level)


_configure_file_logging()
