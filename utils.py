import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_dataarray_map(dataarray, **plot_kws):
    plt.figure(figsize=(10, 5))

    p = dataarray.plot.contourf(
        subplot_kws=dict(projection=ccrs.PlateCarree()),
        transform=ccrs.PlateCarree(),
        **plot_kws
    )

    p.axes.coastlines()
    gl = p.axes.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        alpha=0.3,
    )
    gl.top_labels = False
    gl.right_labels = False
    return p


def munich_station_monthly(location=".", to_xarray=False):
    """
    Open DWD munich station with pandas
    :param location: (str) absolute or relative location produkt_klima_monat_19540601_20181231_03379.txt file. No slash at end.
    :param to_xarray: (boolean) if True return xr.Dataset with variables t_mean, t_max, t_min. if False return full pandas DataFrame
    :return: (pd.DataFrame or xr.Dataset) DataFrame of DWD Munich station
    """
    path = location + "/produkt_klima_monat_19540601_20181231_03379.txt"
    t_monthly_raw = pd.read_table(
        path,
        sep=";",  # columns are separated by semicolons
        date_parser=lambda x: datetime.strptime(x, "%Y%m%d"),  # specify date format
        parse_dates=[1, 2, ],  # MESS_DATUM_BEGINN and MESS_DATUM_ENDE are dates
        index_col="MESS_DATUM_BEGINN",  # set index
    ).rename(
        columns=lambda x: x.strip()  # removes header white spaces, e.g. " MO_TT" -> "MO_TT"
    )
    if to_xarray:
        t_monthly = (
            t_monthly_raw[["MO_TT", "MO_TX", "MO_TN"]]  # pick only temperature columns
                .to_xarray()  # convert pandas DataFrame to xarray Dataset
                .rename(MESS_DATUM_BEGINN="time", MO_TT="t_mean", MO_TX="t_max", MO_TN="t_min")
        )
        return t_monthly
    else:
        return t_monthly_raw
    
    
    
def munich_station_daily(location=".", to_xarray=False):
    """
    Open DWD munich station with pandas
    :param location: (str) absolute or relative location produkt_klima_monat_19540601_20181231_03379.txt file. No slash at end.
    :param to_xarray: (boolean) if True return xr.Dataset with variables t_mean, t_max, t_min. if False return full pandas DataFrame
    :return: (pd.DataFrame or xr.Dataset) DataFrame of DWD Munich station
    """
    path = location + "/produkt_klima_tag_19540601_20191231_03379.txt"
    t_daily_raw = pd.read_table(
        path,
        sep=";",  # columns are separated by semicolons
        date_parser=lambda x: datetime.strptime(x, "%Y%m%d"),  # specify date format
        parse_dates=[1],  # MESS_DATUM dates
        index_col="MESS_DATUM",  # set index
    ).rename(
        columns=lambda x: x.strip()  # removes header white spaces, e.g. " MO_TT" -> "MO_TT"
    )
    if to_xarray:
        t_daily = (
            t_daily_raw[["TMK", "TXK", "TNK", "RSK"]]  # pick temperature and precipitation columns
                .to_xarray()  # convert pandas DataFrame to xarray Dataset
                .rename(MESS_DATUM = "time", TMK="t_mean", TXK="t_max", TNK="t_min", RSK="precip")
        )
        return t_daily
    else:
        return t_daily_raw


def berlin_station_monthly(location=".", to_xarray=False):
    data_monthly_raw = pd.read_table(
        location + "/produkt_klima_monat_17190101_20211231_00403.txt",
        sep=";",
        date_parser=lambda x: datetime.strptime(x, "%Y%m%d"),
        parse_dates=[1],
        index_col="MESS_DATUM_BEGINN",
        na_values=-999,
    ).rename(
        columns=lambda x: x.strip()
    )  # removes header white spaces, e.g. " TMK" -> "TMK"
    if to_xarray:
        data_monthly = (
            data_monthly_raw[
                ["MO_TT", "MO_TX", "MO_TN"]
            ]  # pick only temperature columns
            .to_xarray()
            .rename(
                MESS_DATUM_BEGINN="time", MO_TT="t_mean", MO_TX="t_max", MO_TN="t_min"
            )
        )
        return data_monthly
    else:
        return data_monthly_raw
    
    
def get_nao():
    path = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii"
    raw = pd.read_table(
        path,
        delim_whitespace=True,
        header=None,
        names=["year", "month", "day", "NAO"],
        na_values="*******",
    )

    dates = pd.to_datetime(raw[["year", "month", "day"]], errors="coerce")

    nao = (
        raw[["NAO"]]
        .set_index(dates)
        .squeeze()
        .to_xarray()
        .rename(index="time")
        .dropna("time")
    )
    return nao