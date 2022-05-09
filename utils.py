import cartopy.crs as ccrs
import matplotlib.pyplot as plt


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
