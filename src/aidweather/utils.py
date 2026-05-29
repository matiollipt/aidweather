# SPDX-License-Identifier: Apache-2.0
"""
aidweather.utils
~~~~~~~~~~~~~~~~

Cross-cutting DataFrame utilities for the ``aidweather`` package.

This module provides a single, focused primitive used by downstream packages
such as ``aidfarm``: ``ensure_date_column``, which robustly finds, parses,
and standardises a datetime column in a pandas DataFrame.

Core Features:
- ``ensure_date_column``: Robustly finds a date column by name, from a list
  of candidate names, or from a DatetimeIndex, and returns a copy with a
  guaranteed ``datetime64[ns]`` column.

Example:
    >>> from aidweather.utils import ensure_date_column
    >>> import pandas as pd
    >>> df = pd.DataFrame({"obs_date": ["2023-01-01", "2023-06-15"]})
    >>> cleaned = ensure_date_column(df, name="date", candidates=["obs_date"])
    >>> print(cleaned["date"].dtype)
    datetime64[ns]
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass
class DateColumnOptions:
    """Options for configuring ensure_date_column."""

    inplace: bool = False
    candidates: Iterable[str] | None = None
    index_fallback: bool = True
    normalize: bool = True
    strip_timezone: bool = True


def _find_date_source_column(
    work: pd.DataFrame, name: str, candidates: Iterable[str] | None
) -> str | None:
    if name in work.columns:
        return name
    if candidates is not None:
        for cand in candidates:
            if cand in work.columns:
                return cand
    return None


def _coerce_date_column(
    work: pd.DataFrame,
    name: str,
    src_col: str | None,
    index_fallback: bool,
    candidates: Iterable[str] | None,
) -> pd.DataFrame:
    if src_col is not None:
        work[src_col] = pd.to_datetime(work[src_col], errors="raise")
        work[name] = work[src_col]
        if src_col != name:
            work = work.drop(columns=[src_col])
    elif index_fallback and isinstance(work.index, pd.DatetimeIndex):
        work[name] = work.index
    else:
        raise ValueError(
            f"Could not ensure date column '{name}': neither the column nor "
            f"candidates {list(candidates or [])} were found, and index is "
            "not a DatetimeIndex."
        )
    return work


def _standardize_datetime_column(
    work: pd.DataFrame, name: str, strip_timezone: bool, normalize: bool
) -> pd.DataFrame:
    if strip_timezone:
        work[name] = work[name].dt.tz_localize(None)
    if normalize:
        work[name] = work[name].dt.normalize()
    return work


def ensure_date_column(
    df: pd.DataFrame,
    name: str = "date",
    **kwargs,
) -> pd.DataFrame:
    """Robustly ensures a DataFrame has a datetime column with a specific name.

    Searches for the column by ``name``, then by any ``candidates``, then
    falls back to the DataFrame's DatetimeIndex (when ``index_fallback=True``).
    Returns a copy by default; use ``inplace=True`` to mutate in place.

    Args:
        df: The input DataFrame.
        name: The desired final name for the date column.
        **kwargs: Additional configuration options:
            inplace: If True, modifies the DataFrame in place.
            candidates: A list of alternative column names to search for.
            index_fallback: If True, allows using the DataFrame's index as
                the date source when no matching column is found.
            normalize: If True, normalizes the datetime to midnight.
            strip_timezone: If True, removes timezone information.

    Returns:
        The DataFrame with a guaranteed ``datetime64[ns]`` column named
        ``name``.

    Raises:
        ValueError: If no suitable date column can be found.
    """
    opts = DateColumnOptions(
        inplace=kwargs.get("inplace", False),
        candidates=kwargs.get("candidates", None),
        index_fallback=kwargs.get("index_fallback", True),
        normalize=kwargs.get("normalize", True),
        strip_timezone=kwargs.get("strip_timezone", True),
    )

    work = df if opts.inplace else df.copy()

    src_col = _find_date_source_column(work, name, opts.candidates)
    work = _coerce_date_column(
        work, name, src_col, opts.index_fallback, opts.candidates
    )
    work = _standardize_datetime_column(work, name, opts.strip_timezone, opts.normalize)

    return work
