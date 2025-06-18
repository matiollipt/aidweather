# %% [markdown]
# # aidweather functional tests

# %%
from power_api import PowerAPI
from IPython.display import display

# %%
# locations
lisbon = (38.69490777628757, -9.351965953613986)

# %%
api = PowerAPI(
    loc_name="Lisbon",
    start="2024-01-01",  # type: ignore
    end="2024-12-31",  # type: ignore
    lat=lisbon[0],
    lon=lisbon[1],
)

weather_df = api.get_dataframes()["short_names"]

# Print some information
print(f"Max date: {weather_df['date'].max()}")
print(f"Min date: {weather_df['date'].min()}")
print(f"Unique dates: {len(weather_df['date'].unique())}")
print(f"Weather data shape: {weather_df.shape}")
print(f"Parameters: {"\n -  ".join(weather_df.columns)}")
display(weather_df.head())

# %%
api.plot(df_key="short_names", parameters=["T2M", "RH2M"], dual_axis=True, ma_window=7)

# %%
api.plot_folium_map(zoom=7)

# %%
aggregated = api.aggregate(df_key="short_names", freq="W", agg={"T2M": "mean"})
display(f"Aggregated data shape: {aggregated.shape}", aggregated.head())

