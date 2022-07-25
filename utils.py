import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.patches as mpatches
import xarray as xr
import glob
import skimage.io


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
            t_daily_raw[["TMK", "TXK", "TNK", "RSK", "PM"]]  # pick temperature and precipitation columns
                .to_xarray()  # convert pandas DataFrame to xarray Dataset
                .rename(MESS_DATUM = "time", TMK="t_mean", TXK="t_max", TNK="t_min", RSK="precip", PM = "pressure")
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




def add_box(id, ax, color="k"):
    if id == "med":
        ax.add_patch(
            mpatches.Rectangle(
                xy=[10, 36],
                width=20,
                height=5,
                edgecolor=color,
                fill=False,
                lw=3,
                transform=ccrs.PlateCarree(),
            )
        )
    elif id == "dk":
        ax.add_patch(
            mpatches.Rectangle(
                xy=[2, 50],
                width=13,
                height=10,
                edgecolor=color,
                fill=False,
                lw=3,
                transform=ccrs.PlateCarree(),
            )
        )
    else:
        print("invalid id")
        
        
        
def linear_fit(x, y):
    # y = a + b * x

    sum_dev = 0
    
    for (i,j) in zip(x,y):
        sum_dev += (i-x.mean())*(j-y.mean())
    
    cov_xy = sum_dev/len(x)
        
    b = cov_xy/x.std()**2
    a = y.mean()-b*x.mean()

    return a, b


def pearson_r (x,y):
    a, b = linear_fit(x,y)
    r = b*x.std()/y.std()
    return r


def munich_station_10min(location="./", to_xarray=False):
    # open file

    data_10min_raw = pd.read_table(
        location + "/produkt_zehn_min_tu_20181004_20200405_03379.txt",
        sep=";",
        date_parser=lambda x: datetime.strptime(x, "%Y%m%d%H%M"),
        parse_dates=[1],
        index_col="MESS_DATUM",
        skipfooter=1,
        engine="python",
    ).rename(
        columns=lambda x: x.strip()
    )  # removes header white spaces
    
    if to_xarray:
        # convert to xarray
        data_10min = data_10min_raw[["TT_10"]].to_xarray().rename(MESS_DATUM="time", TT_10="t")
        return data_10min
    else:
        data_10min_raw
        
        
def get_qbo():
    path = "https://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/qbo.dat"
    dateparse = lambda x: datetime.strptime(x, "%y%m")
    qbo = pd.read_table(
        path,
        skiprows=381,
        delim_whitespace=True,
        index_col=0,
        usecols=range(1, 9),
        names=[
            "IIIII",
            "time",
            "70hPa",
            "50hPa",
            "40hPa",
            "30hPa",
            "20hPa",
            "15hPa",
            "10hPa",
        ],
        date_parser=dateparse,
        parse_dates=[0],
    )
    qbo_xr = qbo.to_xarray()
    data = (
        xr.concat([qbo_xr[v] / 10 for v in qbo_xr.data_vars], dim="p")
        .assign_coords(p=[int(n[:2]) for n in list(qbo_xr.data_vars)])
        .rename("u")
        .assign_attrs(units="m/s")
    )
    data["p"] = data.p.assign_attrs(units="hPa", long_name='pressure level')
    
    return data


def create_signal(
    amplitude_and_period, white_noise_amplitude=0, red_noise_amplitude_and_r1=(0, 0.95)
):
    """
    create a random signal that consists of a harmonic wave, red noise and white noise
    :param amplitude_and_period: list of tuples of floats [(amplitude, period), (amplitude2, period2), ...]
    :param white_noise_amplitude: amplitude of white noise (float)
    :param red_noise_amplitude_and_r1: tuple of floats (amplitude, lag-1-autocorr)
    :return: x, y (np.array, np.array)
    """

    # create grid and empty signal
    x = np.linspace(0, 200, 1_001)
    n = len(x)
    signal = np.zeros(x.shape)

    # waves
    waves = []
    for A, T in amplitude_and_period:
        print("wave: amplitude={}, period={}".format(A, T))
        wave = A * np.cos(2 * np.pi / T * x)
        waves.append(wave)
        signal = signal + wave

    # white noise
    signal = signal + white_noise_amplitude * np.random.normal(size=n)

    # red noise
    rn_A, rn_r1 = red_noise_amplitude_and_r1
    signal = signal + rn_A * ar1_process(n_samples=n, corr=rn_r1)

    # plot signal
    fig, ax = plt.subplots()
    _ = [ax.plot(x, w, alpha=0.8, lw=2.5) for w in waves]
    ax.plot(x, signal, zorder=-2, c="k", lw=1)
    ax.axhline(0, c="k", lw=0.5, zorder=-3)
    ax.set_xlabel("time [s]")
    ax.set_title("Artificial Signal")

    return x, signal


def ar1_process(n_samples, corr, mu=0, sigma=1):
    assert 0 < corr < 1, "Auto-correlation must be between 0 and 1"

    c = mu * (1 - corr)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c + np.random.normal(0, sigma_e)]
    for _ in range(1, n_samples):
        signal.append(c + corr * signal[-1] + np.random.normal(0, sigma_e))

    return np.array(signal)



def read_satellite_images_ger(path):
    """
    Read  satellite images by NASA (https://wvs.earthdata.nasa.gov/?LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&TIME=2000-03-01&COORDINATES=47.000000,5.000000,55.000000,15.000000&FORMAT=image/jpeg&AUTOSCALE=TRUE&RESOLUTION=10km) over Germany and convert into xarray.
    """
    n_could_not_open = 0
    n_weird_shape = 0
    all_files = glob.glob(path)
    dates = []
    imgs = []
    n = len(all_files)
    print(f"reading {n} images...")

    for f in all_files:
        try:
            img = skimage.io.imread(f)
        except:
            n_could_not_open += 1
        else:
            if img.shape != (91, 72, 3):
                n_weird_shape += 1
            else:
                imgs.append(img)
                
                dates.append(f.split(".")[-2].split("_")[-1])

    imgs = np.array(imgs)
    dates = pd.DatetimeIndex(dates)

    print(
    f"Summary:\n   Could not open {n_could_not_open} files (={n_could_not_open / n:.3%})\n   {n_weird_shape} (={n_weird_shape / n:.3%}) images had a weird shape and were skipped"
    )
        
    imgs_xr = xr.DataArray(
        imgs,
        dims=["time", "x", "y", "rgb"],
        coords=dict(time=dates, x=range(91), y=range(72), rgb=["r", "g", "b"]),
        name="imgs",
    )
    return imgs_xr.sortby("time")


def munich_berlin_sylt_station_yearly(path):
    # open file

    locations = ["berlin", "munich", "sylt"]
    ds_list = []

    for loc in locations:
        df = pd.read_table(
            f"{path}/{loc}.txt",
            sep=";",
            date_parser=lambda x: datetime.strptime(x, "%Y%m%d"),
            parse_dates=[1],
            index_col="MESS_DATUM_BEGINN",
            skipfooter=1,
            engine="python",
            na_values=-999,
        ).rename(
            columns=lambda x: x.strip()
        )  # removes header white spaces
        # convert to xarray
        ds = (
            df[["JA_MX_FX", "JA_RR", "JA_TT"]]  # JA_MX_FX
            .to_xarray()
            .rename(MESS_DATUM_BEGINN="time", JA_MX_FX="wind", JA_RR="precip", JA_TT="temp")
        )
        ds = ds.astype("float").assign_coords(location=loc)
        ds_list.append(ds)
    data = xr.concat(ds_list, dim="location")
    return data