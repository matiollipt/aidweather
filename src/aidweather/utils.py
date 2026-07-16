# SPDX-License-Identifier: Apache-2.0
"""
aidweather.utils
~~~~~~~~~~~~~~~~

DataFrame date-column utilities.

Provides ``ensure_date_column``, which locates, parses, and standardises a
datetime column in a pandas DataFrame by name, candidate list, or DatetimeIndex.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

__all__ = ["ensure_date_column", "DateColumnOptions"]


@dataclass(frozen=True)
class DateColumnOptions:
    """Options for configuring ``ensure_date_column``."""

    inplace: bool = False
    candidates: Iterable[str] | None = None
    index_fallback: bool = True
    normalize: bool = False
    strip_timezone: bool = True


def _find_date_source_column(
    work: pd.DataFrame, name: str, candidates: Iterable[str] | None
) -> str | None:
    """Return the first of *name* or *candidates* present in *work*'s columns, else ``None``."""
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
    """Parse *src_col* (or the DatetimeIndex) into *name*, raising if neither is available."""
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
    """Strip timezone info and/or normalize *name* to midnight, per the given flags."""
    if strip_timezone and work[name].dt.tz is not None:
        # tz_convert(None) strips the timezone and converts to UTC-equivalent naive
        # timestamps.  tz_localize(None) would raise TypeError on tz-naive columns.
        work[name] = work[name].dt.tz_convert(None)
    if normalize:
        work[name] = work[name].dt.normalize()
    return work


def ensure_date_column(
    df: pd.DataFrame,
    name: str = "date",
    *,
    inplace: bool = False,
    candidates: Iterable[str] | None = None,
    index_fallback: bool = True,
    normalize: bool = False,
    strip_timezone: bool = True,
) -> pd.DataFrame:
    """Ensure *df* has a ``datetime64[ns]`` column named *name*.

    Searches by *name*, then any *candidates*, then falls back to a
    DatetimeIndex. Returns a copy by default; pass ``inplace=True`` to mutate.

    Args:
        df: Input DataFrame.
        name: Target column name for the datetime column.
        inplace: If ``True``, mutate *df* in place; otherwise return a copy.
        candidates: Alternative column names to search if *name* is not found.
        index_fallback: If ``True``, extract dates from a ``DatetimeIndex`` when
            no suitable column is found.
        normalize: If ``True``, floor all times to midnight after parsing.
        strip_timezone: If ``True``, strip timezone info to produce tz-naive timestamps.

    Raises:
        ValueError: If no suitable date source is found.
    """
    opts = DateColumnOptions(
        inplace=inplace,
        candidates=candidates,
        index_fallback=index_fallback,
        normalize=normalize,
        strip_timezone=strip_timezone,
    )

    work = df if opts.inplace else df.copy()

    src_col = _find_date_source_column(work, name, opts.candidates)
    work = _coerce_date_column(
        work, name, src_col, opts.index_fallback, opts.candidates
    )
    work = _standardize_datetime_column(work, name, opts.strip_timezone, opts.normalize)

    return work
