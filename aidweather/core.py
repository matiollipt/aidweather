import requests
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Union, Tuple
from aidweather.config import cfg

# load default config (NASA POWER API + default params)
BASE_URL = cfg["base_url"]
WEATHER_PARAMS_DEFAULT = cfg["weather_params_default"]


class AidWeather:
    """
    Client for NASA POWER daily weather data with flexible plotting and cross-analysis features.
    """

    BASE_URL = cfg["base_url"]
    WEATHER_PARAMS_DEFAULT = cfg["weather_params_default"]

    def __init__(
        self,
        name: str,
        lat: float,
        lon: float,
        start: Union[str, datetime],
        end: Union[str, datetime],
        params: Optional[List[str]] = None,
        session: Optional[requests.Session] = None,
    ):
        """
        Initialize a AidWeather client.
        """
        self.name = name
        self.lat = lat
        self.lon = lon
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.params = params or list(self.WEATHER_PARAMS_DEFAULT.keys())
        self.session = session or requests.Session()
        self._df: Optional[pd.DataFrame] = None

    def _build_url(self) -> str:
        pstr = ",".join(self.params)
        return (
            f"{self.BASE_URL}?parameters={pstr}"  # type: ignore
            f"&community=RE&latitude={self.lat}&longitude={self.lon}"
            f"&start={self.start:%Y%m%d}&end={self.end:%Y%m%d}&format=JSON"
        )

    def _fetch(self) -> Dict[str, Dict[str, float]]:
        resp = self.session.get(self._build_url())
        resp.raise_for_status()
        data = resp.json().get("properties", {}).get("parameter", {})
        if not data:
            raise ValueError("No data returned from NASA POWER API.")
        return data

    def _to_df(self, raw: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(raw)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "date"
        return df.reset_index()

    def load_df(self) -> pd.DataFrame:
        """
        Fetch and cache raw data as DataFrame with short column names.
        """
        if self._df is None:
            raw = self._fetch()
            self._df = self._to_df(raw)
        return self._df.copy()

    def get(self, short_names: bool = True) -> pd.DataFrame:
        """
        Return DataFrame with optional descriptive column names.
        """
        df = self.load_df()
        return df.rename(columns=self._name_map()) if not short_names else df

    def _name_map(self) -> Dict[str, str]:
        return {k: self.WEATHER_PARAMS_DEFAULT.get(k, k) for k in self.params}

    def filter(
        self,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> pd.DataFrame:
        """
        Filter by date range and value thresholds.
        """
        df = self.get(short_names=False)
        if start or end:
            s = pd.to_datetime(start) if start else df.date.min()
            e = pd.to_datetime(end) if end else df.date.max()
            df = df[df.date.between(s, e)]
        if thresholds:
            for col, (lo, hi) in thresholds.items():
                df = df[df[col].between(lo, hi)]
        return df

    def aggregate(
        self,
        freq: str = "M",
        agg: Union[str, Dict[str, Union[str, List[str]]]] = "mean",
    ) -> pd.DataFrame:
        """
        Aggregated weather data at given frequency.
        """
        return (
            self.get(short_names=False)
            .set_index("date")
            .resample(freq)
            .agg(agg)
            .reset_index()
        )
