# aidweather/dataviz.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from typing import Optional, List, Tuple, Union
from datetime import datetime


class Visualizer:
    def __init__(self, aidweather, font_scale: Optional[float] = None):
        self.client = aidweather
        self.name = aidweather.name
        self.WEATHER_PARAMS = aidweather.WEATHER_PARAMS_DEFAULT
        self.font_scale = font_scale

    def _apply_font_scaling(self, fig: plt.Figure, base_size: float = 12) -> None:
        w, h = fig.get_size_inches()
        factor = self.font_scale if self.font_scale else min(w, h) / 6
        size = base_size * factor
        plt.rcParams.update(
            {
                "axes.titlesize": size * 1.1,
                "axes.labelsize": size,
                "xtick.labelsize": size * 0.8,
                "ytick.labelsize": size * 0.8,
                "legend.fontsize": size * 0.9,
            }
        )

    def plot(
        self,
        cols: Optional[List[str]] = None,
        ma: int = 0,
        dual: bool = False,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        date_range: Optional[Tuple[Union[str, datetime], Union[str, datetime]]] = None,
    ) -> None:
        df = self.client.load_df()
        if date_range:
            s, e = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[df.date.between(s, e)]

        cols = cols or [c for c in df.columns if c != "date"]
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Columns not found: {missing}")

        fig, ax = plt.subplots(figsize=figsize)
        self._apply_font_scaling(fig)
        ax2 = ax.twinx() if dual and len(cols) > 1 else None
        cmap = plt.cm.get_cmap("Dark2", len(cols))

        for i, code in enumerate(cols):
            srs = df[["date", code]].copy()
            if ma:
                srs[code] = srs[code].rolling(ma).mean()
            label = self.WEATHER_PARAMS.get(code, code)
            tgt_ax = ax2 if dual and i == 1 else ax
            tgt_ax.plot(srs.date, srs[code], label=label, color=cmap(i))

        ax.set_xlabel("Date")
        ax.set_ylabel(
            "Value" if len(cols) > 1 else self.WEATHER_PARAMS.get(cols[0], cols[0])
        )
        ax.set_title(
            title
            or f"{self.name}: {self.client.start.date()} to {self.client.end.date()}"
        )
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
        plt.show()

    def overlay(
        self,
        other: pd.DataFrame,
        other_cols: List[str],
        weather_cols: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        date_range: Optional[Tuple[Union[str, datetime], Union[str, datetime]]] = None,
        agri_ma: int = 0,  # <-- rolling window for agri
        weather_ma: int = 0,  # <-- rolling window for weather
        agri_kind: str = "errorbar",
        weather_kind: str = "line",
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        # — prepare agri: apply rolling if requested —
        df_ag = other.copy()
        df_ag["date"] = pd.to_datetime(df_ag["date"])
        df_ag.set_index("date", inplace=True)
        if agri_ma > 1:
            df_ag = df_ag[other_cols].rolling(agri_ma, center=True).mean().dropna()
            # turn back into a “date” column for our stats helper
            df_ag = df_ag.reset_index()

        stats = self._prepare_agri_stats(df_ag, other_cols)

        # — prepare weather: same idea —
        df_w = self._prepare_weather(weather_cols)
        if weather_ma > 1:
            df_w = df_w.rolling(weather_ma, center=True).mean().dropna()

        # — date‐filter if asked —
        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            stats = stats.loc[start:end]
            df_w = df_w.loc[start:end]

        # — plotting —
        fig, ax1 = plt.subplots(figsize=figsize)
        self._apply_font_scaling(fig)
        ax2 = ax1.twinx()

        ag_h, ag_l = self._plot_agri(ax1, stats, other_cols, kind=agri_kind)
        w_h, w_l = self._plot_weather(ax2, df_w, weather_cols or [], kind=weather_kind)

        self._create_legends(ax1, ax2, ag_h, ag_l, w_h, w_l)
        self._finalize(
            fig, ax1, title or f"{self.name} Crop/Weather Overlay", save_path
        )

        return fig, ax1, ax2

    def _plot_agri(
        self,
        ax,
        stats: pd.DataFrame,
        cols: List[str],
        kind: str = "errorbar",
    ) -> Tuple[List, List]:
        cmap = plt.get_cmap("tab10", len(cols))
        handles, labels = [], []

        if kind == "bar":
            xnum = mdates.date2num(stats.index.to_pydatetime())
            width = (np.min(np.diff(xnum)) if len(xnum) > 1 else 1.0) * 0.8

        for i, c in enumerate(cols):
            x = stats.index
            y = stats[f"{c}_mean"]
            yerr = stats[f"{c}_std"].fillna(0)

            if kind == "errorbar":
                h = ax.errorbar(x, y, yerr=yerr, fmt="o-", color=cmap(i), capsize=4)
            elif kind == "bar":
                h = ax.bar(x, y, width=width, yerr=yerr, color=cmap(i), capsize=4)
            elif kind == "line":
                (h,) = ax.plot(x, y, linestyle="-", color=cmap(i))
            elif kind == "scatter":
                h = ax.scatter(x, y, color=cmap(i), marker="o")
            else:
                raise ValueError(f"Unknown agri kind: {kind}")

            handles.append(h)
            labels.append(c)

        if kind == "bar":
            ax.xaxis_date()
            ax.figure.autofmt_xdate()

        ax.set_ylabel("Measurements (cm, mm)")
        return handles, labels

    def _plot_weather(
        self,
        ax,
        df_w: pd.DataFrame,
        cols: List[str],
        kind: str = "line",
    ) -> Tuple[List, List]:
        cmap = plt.get_cmap("Dark2", len(cols))
        handles, labels = [], []

        if kind == "bar":
            xnum = mdates.date2num(df_w.index.to_pydatetime())
            width = (np.min(np.diff(xnum)) if len(xnum) > 1 else 1.0) * 0.8

        for i, code in enumerate(cols):
            x = df_w.index
            y = df_w[code]

            if kind == "line":
                (h,) = ax.plot(x, y, alpha=0.7, linewidth=1.2, color=cmap(i))
            elif kind == "scatter":
                h = ax.scatter(x, y, alpha=0.7, color=cmap(i), s=20)
            elif kind == "bar":
                h = ax.bar(x, y, width=width, alpha=0.7, color=cmap(i))
            else:
                raise ValueError(f"Unknown weather kind: {kind}")

            handles.append(h)
            labels.append(self.WEATHER_PARAMS.get(code, code))

        if kind == "bar":
            ax.xaxis_date()
            ax.figure.autofmt_xdate()

        ax.set_ylabel(" / ".join(self.WEATHER_PARAMS.get(c, c) for c in cols))
        return handles, labels

    def _prepare_agri_stats(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        grp = df.groupby("date")[cols].agg(["mean", "std"])
        grp.columns = [f"{col}_{stat}" for col, stat in grp.columns]
        return grp

    def _prepare_weather(self, weather_cols: Optional[List[str]]) -> pd.DataFrame:
        df_w = self.client.load_df().copy()
        df_w["date"] = pd.to_datetime(df_w["date"])
        df_w.set_index("date", inplace=True)

        if weather_cols is None:
            weather_cols = list(df_w.columns)
        missing = set(weather_cols) - set(df_w.columns)
        if missing:
            raise KeyError(f"Weather codes not found: {missing}")
        return df_w[weather_cols]

    def _create_legends(
        self,
        ax1,
        ax2,
        ag_handles,
        ag_labels,
        w_handles,
        w_labels,
    ) -> None:
        # Agri legend on ax1
        leg1 = ax1.legend(ag_handles, ag_labels, title="Agri Data", loc="upper left")
        ax1.add_artist(leg1)
        # Weather legend on ax2
        ax2.legend(w_handles, w_labels, title="Weather Data", loc="upper right")

    def _finalize(
        self, fig, ax1, title: Optional[str], save_path: Optional[str]
    ) -> None:
        ax1.set_xlabel("Date")
        ax1.set_title(title or f"{self.name} Crop/Weather Overlay")
        ax1.grid(True)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
        plt.show()

    # Internal helpers (_prepare_agri_stats, _prepare_weather, _plot_agri, _plot_weather,
    # _create_legends, _finalize remain unchanged except for fontsize parameters handled globally)
