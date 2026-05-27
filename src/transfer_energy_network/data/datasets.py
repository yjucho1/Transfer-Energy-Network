"""Dataset registry for long-term forecasting benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    file_name: str
    target: str
    freq: str
    num_features: int
    description: str


LTSF_DATASETS: dict[str, DatasetProfile] = {
    "ETTh1": DatasetProfile(
        name="ETTh1",
        file_name="ETT-small/ETTh1.csv",
        target="OT",
        freq="h",
        num_features=7,
        description="Electricity Transformer Temperature, hourly subset 1",
    ),
    "ETTh2": DatasetProfile(
        name="ETTh2",
        file_name="ETT-small/ETTh2.csv",
        target="OT",
        freq="h",
        num_features=7,
        description="Electricity Transformer Temperature, hourly subset 2",
    ),
    "ETTm1": DatasetProfile(
        name="ETTm1",
        file_name="ETT-small/ETTm1.csv",
        target="OT",
        freq="15min",
        num_features=7,
        description="Electricity Transformer Temperature, 15-minute subset 1",
    ),
    "ETTm2": DatasetProfile(
        name="ETTm2",
        file_name="ETT-small/ETTm2.csv",
        target="OT",
        freq="15min",
        num_features=7,
        description="Electricity Transformer Temperature, 15-minute subset 2",
    ),
    "Electricity": DatasetProfile(
        name="Electricity",
        file_name="electricity/electricity.csv",
        target="OT",
        freq="h",
        num_features=321,
        description="Electricity consumption benchmark used in LTSF studies",
    ),
    "Traffic": DatasetProfile(
        name="Traffic",
        file_name="traffic/traffic.csv",
        target="OT",
        freq="h",
        num_features=862,
        description="Road occupancy benchmark used in LTSF studies",
    ),
    "Weather": DatasetProfile(
        name="Weather",
        file_name="weather/weather.csv",
        target="OT",
        freq="10min",
        num_features=21,
        description="Weather benchmark used in LTSF studies",
    ),
    "Exchange": DatasetProfile(
        name="Exchange",
        file_name="exchange_rate/exchange_rate.csv",
        target="OT",
        freq="d",
        num_features=8,
        description="Exchange rate benchmark used in LTSF studies",
    ),
    "ILI": DatasetProfile(
        name="ILI",
        file_name="illness/national_illness.csv",
        target="OT",
        freq="w",
        num_features=7,
        description="Influenza-like illness benchmark used in LTSF studies",
    ),
    "etth1": DatasetProfile(
        name="etth1",
        file_name="ETT-small/ETTh1.csv",
        target="OT",
        freq="h",
        num_features=7,
        description="ProbTS key for Electricity Transformer Temperature, hourly subset 1",
    ),
    "etth2": DatasetProfile(
        name="etth2",
        file_name="ETT-small/ETTh2.csv",
        target="OT",
        freq="h",
        num_features=7,
        description="ProbTS key for Electricity Transformer Temperature, hourly subset 2",
    ),
    "ettm1": DatasetProfile(
        name="ettm1",
        file_name="ETT-small/ETTm1.csv",
        target="OT",
        freq="15min",
        num_features=7,
        description="ProbTS key for Electricity Transformer Temperature, 15-minute subset 1",
    ),
    "ettm2": DatasetProfile(
        name="ettm2",
        file_name="ETT-small/ETTm2.csv",
        target="OT",
        freq="15min",
        num_features=7,
        description="ProbTS key for Electricity Transformer Temperature, 15-minute subset 2",
    ),
    "electricity_ltsf": DatasetProfile(
        name="electricity_ltsf",
        file_name="electricity/electricity.csv",
        target="OT",
        freq="h",
        num_features=321,
        description="ProbTS key for the Electricity long-term forecasting benchmark",
    ),
    "traffic_ltsf": DatasetProfile(
        name="traffic_ltsf",
        file_name="traffic/traffic.csv",
        target="OT",
        freq="h",
        num_features=862,
        description="ProbTS key for the Traffic long-term forecasting benchmark",
    ),
    "weather_ltsf": DatasetProfile(
        name="weather_ltsf",
        file_name="weather/weather.csv",
        target="OT",
        freq="10min",
        num_features=21,
        description="ProbTS key for the Weather long-term forecasting benchmark",
    ),
    "exchange_ltsf": DatasetProfile(
        name="exchange_ltsf",
        file_name="exchange_rate/exchange_rate.csv",
        target="OT",
        freq="d",
        num_features=8,
        description="ProbTS key for the Exchange Rate long-term forecasting benchmark",
    ),
    "illness_ltsf": DatasetProfile(
        name="illness_ltsf",
        file_name="illness/national_illness.csv",
        target="OT",
        freq="w",
        num_features=7,
        description="ProbTS key for the Illness long-term forecasting benchmark",
    ),
}


def get_dataset_profile(name: str) -> DatasetProfile:
    if name not in LTSF_DATASETS:
        raise KeyError(f"Unknown LTSF dataset: {name}")
    return LTSF_DATASETS[name]
