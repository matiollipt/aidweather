# SPDX-License-Identifier: Apache-2.0
"""
aidweather
~~~~~~~~~~

Weather data retrieval for agricultural and environmental applications.

Exposes ``PowerClient`` (NASA POWER API wrapper), ``GeoCoordinate`` (coordinate
utilities), and ``cfg`` (package configuration).

License:
    Distributed under the Apache-2.0 license.
"""

from __future__ import annotations

__version__ = "0.1.3"
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
    """Attach a FileHandler to the package logger if ``logging_config.enabled`` is true in config.json."""
    log_settings = cfg.logging_config()
    if not log_settings.get("enabled"):
        return

    log_path = log_settings["filename"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    level = getattr(
        logging, str(log_settings.get("level", "INFO")).upper(), logging.INFO
    )
    handler = logging.FileHandler(log_path)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    _logger.addHandler(handler)
    _logger.setLevel(level)


_configure_file_logging()
