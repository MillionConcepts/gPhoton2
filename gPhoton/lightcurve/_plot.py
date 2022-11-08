from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TITILLIUM = Path(
    Path(__file__).parent, "static/fonts", "TitilliumWeb-Light.ttf"
)
TITLE_FONT = mpl.font_manager.FontProperties(fname=TITILLIUM, size=10)


def make_source_figs(
    source_table: pd.DataFrame,
    segment_map: np.ndarray,
    cnt_image: np.ndarray,
    eclipse,
    band: str,
    leg: int,
    outpath=".",
):
    mpl.use("agg")
    # segmentation map
    fig = fig_plot(segment_map, f"e{eclipse}_{band}_segmented")
    prefix = f"e{str(eclipse).zfill(5)}-{band[0].lower()}d-"
    fig.savefig(
        Path(outpath, f"{prefix}segmentation_leg{leg}.jpg"),
        bbox_inches="tight",
        pad_inches=0.1
    )
    # extended source map
    #fig = fig_plot(extended_source_mask, f"e{eclipse}_{band}_extended_mask")
    #fig.savefig(
    #    Path(outpath, f"{prefix}extended-mask.jpg"),
    #    bbox_inches="tight",
    #    pad_inches=0.1
    #)
    # sources plotted on eclipse as circles
    fig = fig_plot_sources(
        cnt_image, source_table, f"e{eclipse}_{band}_{leg}_sources",
    )
    fig.savefig(
        Path(outpath, f"{prefix}-leg{leg}-sources-on-image.jpg"),
        bbox_inches="tight",
        pad_inches=0.1
    )


def shrink_ticks(ax, fontsize=4.5):
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_fontsize(fontsize)
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_fontsize(fontsize)


def fig_plot(array, name):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    cmap = plt.get_cmap('viridis', np.max(array) - np.min(array) + 1)
    cmap.set_under(color='white')
    vmin = 0 if np.max(array) == 0 else 0.5
    plt.imshow(array, cmap=cmap, vmin=vmin)
    ax.set_title(name, fontproperties=TITLE_FONT)
    shrink_ticks(ax)
    fig.tight_layout()
    return fig


def fig_plot_sources(array, sources, name):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.imshow(centile_clip(array), cmap='viridis', interpolation='none')
    ax.set_title(name, fontproperties=TITLE_FONT)
    #points = sources.dropna()
    points = sources
    points.to_csv(f"{name}_pts.csv")
    if "extended_source" in points:
        n_extended = len(points['extended_source'].unique())
        cmap = plt.get_cmap('tab20', n_extended) # was - 1
        cmap.set_under(color='white')
        cmap((points['extended_source'] - 1) / (n_extended - 1))
        #if n_extended > 1:
        point_color = cmap((points['extended_source']) / (n_extended))
        #else:
        #    point_color = 'red'
    else:
        point_color = 'red'
    point_size = points['equivalent_radius']
    plt.scatter(
        points['xcentroid'],
        points['ycentroid'],
        facecolors='none',
        edgecolors=point_color,
        s=point_size,
        lw=.75
    )
    ax.set_title(name, fontproperties=TITLE_FONT)
    shrink_ticks(ax)
    fig.tight_layout()
    return fig


def centile_clip(image, centiles=(0, 90)):
    finite = np.ma.masked_invalid(image)
    bounds = np.percentile(finite[~finite.mask].data, centiles)
    result = np.ma.clip(finite, *bounds)
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data
