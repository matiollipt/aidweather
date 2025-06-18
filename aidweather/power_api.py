import requests
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Dict
from datetime import date, datetime
from sklearn.linear_model import LinearRegression


class PowerAPI:
    """
    Query the NASA POWER API for daily weather data.
    https://power.larc.nasa.gov/docs/services/api/temporal/daily/
    """

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point?"

    DEFAULT_PARAMS = [
        "T2M_RANGE",
        "TS",
        "T2MDEW",
        "T2MWET",
        "T2M_MAX",
        "T2M_MIN",
        "T2M",
        "QV2M",
        "RH2M",
        "PRECTOTCORR",
        "PS",
        "WS10M",
        "WS10M_MAX",
        "WS10M_MIN",
        "WS10M_RANGE",
        "WS50M",
        "WS50M_MAX",
        "WS50M_MIN",
        "WS50M_RANGE",
    ]

    PARAMETER_DESCRIPTIONS = {
        "T2M": "Temperature at 2 Meters (°C)",
        "T2M_MAX": "Max Temperature at 2 Meters (°C)",
        "T2M_MIN": "Min Temperature at 2 Meters (°C)",
        "PRECTOTCORR": "Precipitation Corrected (mm/day)",
        "RH2M": "Rel. Humidity at 2 Meters (%)",
        "WS10M": "Wind Speed at 10 Meters (m/s)",
        "WS10M_MAX": "Max Wind Speed at 10 Meters (m/s)",
        "WS10M_MIN": "Min Wind Speed at 10 Meters (m/s)",
        "WS10M_RANGE": "Wind Speed Range at 10 Meters (m/s)",
        "WS50M": "Wind Speed at 50 Meters (m/s)",
        "WS50M_MAX": "Max Wind Speed at 50 Meters (m/s)",
        "WS50M_MIN": "Min Wind Speed at 50 Meters (m/s)",
        "WS50M_RANGE": "Wind Speed Range at 50 Meters (m/s)",
        "T2MDEW": "Dew Point Temperature at 2 Meters (°C)",
        "T2MWET": "Wet Bulb Temperature at 2 Meters (°C)",
        "TS": "Earth Skin Temperature (°C)",
        "T2M_RANGE": "Temperature Range at 2 Meters (°C)",
        "QV2M": "Specific Humidity at 2 Meters (kg/kg)",
        "PS": "Surface Pressure (kPa)",
    }

    def __init__(
        self,
        loc_name: str,
        start: Union[date, datetime, pd.Timestamp],
        end: Union[date, datetime, pd.Timestamp],
        lat: float,
        lon: float,
        parameter: Optional[List[str]] = None,
    ):
        self.loc_name = loc_name
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.lat = lat
        self.lon = lon
        self.parameter = parameter or self.DEFAULT_PARAMS
        self._request_url = self._build_request()

    def _build_request(self) -> str:
        params = ",".join(self.parameter)
        return (
            f"{self.BASE_URL}parameters={params}"
            f"&community=RE&longitude={self.lon}&latitude={self.lat}"
            f"&start={self.start.strftime('%Y%m%d')}"
            f"&end={self.end.strftime('%Y%m%d')}"
            f"&format=JSON"
        )

    def get_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data and return two DataFrames:
        - 'short_names': raw parameter names
        - 'long_names': parameter names with descriptions
        """
        r = requests.get(self._request_url)
        if r.status_code != 200:
            raise RuntimeError(f"Request failed: {r.status_code} - {r.text}")

        data = r.json().get("properties", {}).get("parameter", {})
        if not data:
            raise ValueError("No data returned.")

        # Reshape
        by_date = {}
        for param, series in data.items():
            for day, val in series.items():
                by_date.setdefault(day, {})[param] = val

        df = pd.DataFrame.from_dict(by_date, orient="index")
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)

        short_names = df.copy()
        long_names = df.rename(columns=self._get_full_feature_names())

        return {"short_names": short_names, "long_names": long_names}

    def _get_full_feature_names(self) -> Dict[str, str]:
        """Create mapping from raw parameter name to snake_case with description."""
        return {k: self.PARAMETER_DESCRIPTIONS.get(k, k) for k in self.parameter}

    def plot(
        self,
        df_key: str = "long_names",
        parameters: Optional[List[str]] = None,
        figsize: tuple = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        ma_window: Optional[int] = None,
        dual_axis: bool = False,
    ):
        """
        Plot parameters over time with optional moving average and dual axis.

        Parameters
        ----------
        df_key : str
            'short_names' or 'long_names'.
        parameters : list of str, optional
            Columns to plot.
        figsize : tuple
            Size of the plot.
        title : str, optional
            Plot title.
        save_path : str, optional
            If provided, saves the plot to this file.
        ma_window : int, optional
            Window size for moving average.
        dual_axis : bool
            If True and two parameters are selected, plots the second on a twin y-axis.
        """
        dfs = self.get_dataframes()
        data = dfs.get(df_key)
        if data is None:
            raise ValueError("Invalid df_key. Use 'short_names' or 'long_names'.")

        if parameters is None:
            parameters = [col for col in data.columns if col != "date"]

        if not parameters or any(p not in data.columns for p in parameters):
            raise ValueError("Invalid or missing parameters.")

        plt.figure(figsize=figsize)
        ax = plt.gca()

        for i, param in enumerate(parameters):
            series = data[["date", param]].copy()
            if ma_window:
                series[param] = series[param].rolling(ma_window).mean()

            if dual_axis and i == 1:
                ax2 = ax.twinx()
                ax2.plot(
                    series["date"],
                    series[param],
                    label=param,
                    linestyle="--",
                    color="orange",
                )
                ax2.set_ylabel(param, color="orange")
                ax2.tick_params(axis="y", colors="orange")
            else:
                ax.plot(series["date"], series[param], label=param)

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        plt.title(
            title
            or f"NASA POWER Time Series ({self.loc_name} | {self.lat}, {self.lon})\n{self.start.date()} to {self.end.date()})"
        )
        ax.grid(True)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.tight_layout()
        plt.show()

    def filter(
        self,
        df_key: str = "long_names",
        date_range: Optional[tuple] = None,
        threshold: Optional[Dict[str, tuple]] = None,
    ) -> pd.DataFrame:
        """
        Filter data by date range and value thresholds.

        Parameters
        ----------
        df_key : str
            'short_names' or 'long_names'.
        date_range : tuple of (str, str), optional
            Filter between start and end date.
        threshold : dict, optional
            E.g. {"Temperature at 2 Meters (°C)": (20, 30)}

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        dfs = self.get_dataframes()
        df = dfs.get(df_key).copy()  # type: ignore

        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df["date"] >= start) & (df["date"] <= end)]

        if threshold:
            for col, (min_v, max_v) in threshold.items():
                if col in df.columns:
                    df = df[(df[col] >= min_v) & (df[col] <= max_v)]

        return df

    def aggregate(
        self,
        df_key: str = "long_names",
        freq: str = "ME",
        agg: Union[str, Dict[str, Union[str, List[str]]]] = "mean",
    ) -> pd.DataFrame:
        """
        Aggregate the weather data to weekly or monthly resolution.

        Parameters
        ----------
        df_key : str
            'short_names' or 'long_names'.
        freq : str
            Resampling frequency: 'W' for weekly, 'ME' for monthly.
        agg : str or dict
            Aggregation method(s): e.g., 'mean', 'sum', or a dict like {'T2M': 'max'}

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with a new 'date' index.
        """
        dfs = self.get_dataframes()
        df = dfs.get(df_key).copy()  # type: ignore
        if df is None:
            raise ValueError("Invalid df_key.")

        df.set_index("date", inplace=True)
        aggregated = df.resample(freq).agg(agg).reset_index()  # type: ignore
        return aggregated

    def plot_overlay(
        self,
        overlay_df: pd.DataFrame,
        overlay_columns: List[str],
        df_key: str = "long_names",
        parameters: Optional[List[str]] = None,
        figsize: tuple = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        ma_window: Optional[int] = None,
        dual_axis: bool = False,
        overlay_labels: Optional[List[str]] = None,
        overlay_colors: Optional[List[str]] = None,
        marker: str = "o",
    ):
        """
        Plot weather data with overlaid growth/phenotype data.

        For numeric (float) columns:
            - Plots line with mean and std bands.
            - Adds linear regression trend.
        For integer (count) columns:
            - Plots total (line) and optional moving average.
        """
        # Load and validate weather data
        dfs = self.get_dataframes()
        weather_df = dfs.get(df_key)
        if weather_df is None:
            raise ValueError("Invalid df_key.")

        if parameters is None:
            parameters = [col for col in weather_df.columns if col != "date"]
        if not parameters or any(p not in weather_df.columns for p in parameters):
            raise ValueError("Invalid weather parameters.")

        if "date" not in overlay_df.columns:
            raise ValueError("overlay_df must contain a 'date' column.")
        for col in overlay_columns:
            if col not in overlay_df.columns:
                raise ValueError(f"Column '{col}' not found in overlay_df.")

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Plot weather
        for i, param in enumerate(parameters):
            series = weather_df[["date", param]].copy()
            if ma_window:
                series[param] = series[param].rolling(ma_window).mean()
            if dual_axis and i == 1:
                ax2 = ax.twinx()
                ax2.plot(
                    series["date"],
                    series[param],
                    label=param,
                    linestyle="--",
                    color="orange",
                )
                ax2.set_ylabel(param, color="orange")
                ax2.tick_params(axis="y", colors="orange")
            else:
                ax.plot(series["date"], series[param], label=param)

        overlay_df = overlay_df.sort_values("date")
        for i, col in enumerate(overlay_columns):
            label = (
                overlay_labels[i] if overlay_labels and i < len(overlay_labels) else col
            )
            color = (
                overlay_colors[i]
                if overlay_colors and i < len(overlay_colors)
                else None
            )

            series = overlay_df[["date", col]].dropna().copy()
            series["ordinal"] = series["date"].map(pd.Timestamp.toordinal)

            if pd.api.types.is_float_dtype(series[col]):
                # Mean line
                ax.plot(series["date"], series[col], label=f"{label}", color=color)
                # Std deviation shading
                rolling_mean = series[col].rolling(ma_window or 3, min_periods=1).mean()
                rolling_std = series[col].rolling(ma_window or 3, min_periods=1).std()
                ax.fill_between(
                    series["date"],
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    color=color,
                    alpha=0.2,
                    label=f"{label} ± std",
                )
                # Regression line
                X = series["ordinal"].values.reshape(-1, 1)
                y = series[col].values.reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                ax.plot(
                    series["date"],
                    y_pred.ravel(),
                    linestyle="--",
                    color=color,
                    alpha=0.7,
                    label=f"{label} (trend)",
                )

            elif pd.api.types.is_integer_dtype(series[col]):
                # Total count line
                ax.plot(series["date"], series[col], label=f"{label}", color=color)
                # Moving average
                if ma_window:
                    ma = series[col].rolling(ma_window, min_periods=1).mean()
                    ax.plot(
                        series["date"],
                        ma,
                        linestyle="--",
                        color=color,
                        alpha=0.7,
                        label=f"{label} (MA{ma_window})",
                    )
            else:
                # Fallback to dots
                ax.scatter(
                    series["date"],
                    series[col],
                    label=label,
                    color=color,
                    marker=marker,
                    zorder=5,
                )

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()
        plt.title(
            title
            or f"NASA POWER with Overlay ({self.loc_name} | {self.lat}, {self.lon})\n{self.start.date()} to {self.end.date()})"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_folium_map(
        self,
        zoom: int = 7,
        args: Optional[List[str]] = None,
    ):

        # Create Folium map
        m = folium.Map(
            location=(self.lat, self.lon),
            zoom_start=zoom,
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Satellite",
            name="Satellite",
            overlay=False,
            control=True,
        ).add_to(m)

        # Add markers

        folium.Marker(
            location=(self.lat, self.lon),
            popup=f"Latitude: {self.lat}, Longitude: {self.lon}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        # Add layer control and display
        folium.LayerControl().add_to(m)

        return m
