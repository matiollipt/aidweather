# SPDX-License-Identifier: Apache-2.0
"""
aidweather
~~~~~~~~~~

Weather data retrieval for agricultural and environmental applications.

Public API
----------
- ``PowerClient``          — NASA POWER API wrapper
- ``GeoCoordinate``        — coordinate container and utilities
- ``normalize_coord_input`` — normalise raw lat/lon inputs
- ``cfg``                  — package configuration singleton
- ``get_config``           — return the ``cfg`` singleton
- ``ensure_date_column``   — standardise a datetime column in a DataFrame

Side effects on import
----------------------
If ``logging_config.enabled`` is ``true`` in ``config.json`` (or the bundled
default), importing this package attaches a ``FileHandler`` to the
``aidweather`` logger. This is intentional and documented behaviour; do not
remove the ``_configure_file_logging()`` call at module level.

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
    log_dir = os.path.dirname(log_path)
    if log_dir:  # dirname is empty when log_path has no directory component
        os.makedirs(log_dir, exist_ok=True)

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
