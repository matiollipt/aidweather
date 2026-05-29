# SPDX-License-Identifier: Apache-2.0
"""
aidweather
~~~~~~~~~~

Weather data retrieval and validation for agricultural applications.

Provides a production-grade client for NASA's POWER API, with a local SQLite cache,
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

__version__ = "0.1.0"
__author__ = "Cleverson Matiolli"
__url__ = "https://github.com/matiollipt/aidweather"

import logging

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

# Prevent "No handler found" warnings if the library user has not configured logging.
logging.getLogger("aidweather").addHandler(logging.NullHandler())
