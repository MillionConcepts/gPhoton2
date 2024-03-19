from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TITILLIUM = Path(
    Path(__file__).parent, "static/fonts", "TitilliumWeb-Light.ttf"
)
TITLE_FONT = mpl.font_manager.FontProperties(fname=TITILLIUM, size=22)
AXIS_LABEL = mpl.font_manager.FontProperties(fname=TITILLIUM, size=18)
TICK_FONT = mpl.font_manager.FontProperties(fname=TITILLIUM, size=14)


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
    fig = fig_plot(segment_map, f"e{eclipse}, {band}")
    prefix = f"e{str(eclipse).zfill(5)}-{band[0].lower()}d-"
    fig.savefig(
        Path(outpath, f"{prefix}{leg:02}-segmap.jpg"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400
    )
    # extended source map
    #fig = fig_plot(extended_source_mask, f"e{eclipse}_{band}_extended_mask")
    #fig.savefig(
    #   Path(outpath, f"{prefix}extended-mask.jpg"),
    #   bbox_inches="tight",
    #   pad_inches=0.1
    #)
    # sources plotted on eclipse as circles
    fig = fig_plot_sources(
        cnt_image, source_table, f"e{eclipse}, {band}, sources",
    )
    fig.savefig(
        Path(outpath, f"{prefix}{leg:02}-sourcemap.jpg"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400

    )
    fig = fig_eclipse_only(
        cnt_image, f"e{eclipse}, {band}",
    )
    fig.savefig(
        Path(outpath, f"{prefix}{leg:02}-image.jpg"),
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=400
    )


def shrink_ticks(ax, fontsize=4.5):
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_fontsize(fontsize)
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_fontsize(fontsize)


def fig_plot(array, name):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=400)
    cmap = plt.get_cmap('prism', np.max(array) - np.min(array) + 1)
    cmap.set_under(color='white')
    vmin = 0 if np.max(array) == 0 else 0.5
    plt.imshow(array, cmap=cmap, vmin=vmin)
    #ax.set_title(name, fontproperties=TITLE_FONT)
    shrink_ticks(ax)
    ax.tick_params(labelsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def fig_eclipse_only(array, name):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=400)
    ax.imshow(centile_clip(array, (0, 98)), cmap='Greys_r', interpolation='none')
    #ax.set_title(name, fontproperties=TITLE_FONT)
    shrink_ticks(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def fig_plot_sources(array, sources, name):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=400)
    ax.imshow(centile_clip(array, (0, 98)), cmap='Greys_r', interpolation='none')
    #ax.set_title(name, fontproperties=TITLE_FONT)
    #points = sources.dropna()
    points = sources
    points.to_csv(f"{name}_pts.csv")
    if "extended_source" in points:
        points_ext = points[points['extended_source']>0]
        points_single = points[points['extended_source']==0]
        point_size_ext = points_ext['equivalent_radius']
        point_size_single = points_single['equivalent_radius']
        n_extended = len(points['extended_source'].unique()-1)
        cmap = plt.get_cmap('tab20', n_extended) # was - 1
        cmap.set_under(color='white')
        cmap((points['extended_source'] - 1) / (n_extended - 1))
        print(n_extended)
        if n_extended == 3:
            point_color = 'blue'
        else:
            point_color = cmap((points_ext['extended_source']) / (n_extended))
        plt.scatter(
            points_ext['xcentroid'],
            points_ext['ycentroid'],
            facecolors='none',
            edgecolors=point_color,
            s=point_size_ext,
            lw=.75
        )
        plt.scatter(
            points_single['xcentroid'],
            points_single['ycentroid'],
            facecolors='none',
            edgecolors="red",
            s=point_size_single,
            lw=.75
        )

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
    #ax.set_title(name, fontproperties=TITLE_FONT)
    shrink_ticks(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def centile_clip(image, centiles=(0, 90)):
    finite = np.ma.masked_invalid(image)
    bounds = np.percentile(finite[~finite.mask].data, centiles)
    result = np.ma.clip(finite, *bounds)
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data
