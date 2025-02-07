import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def npg_palette():
    palette = ["#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF",
               "#8491B4FF", "#91D1C2FF", "#DC0000FF", "#7E6148FF", "#B09C85FF"]
    return sns.color_palette(palette, len(palette))


def jco_palette():
    palette = ["#0073C2FF", "#EFC000FF", "#868686FF", "#CD534CFF", "#7AA6DCFF", "#003C67FF", "#8F7700FF", "#3B3B3BFF",
               "#A73030FF", "#4A6990FF"]
    return sns.color_palette(palette, len(palette))


def get_sns_palette_cycle():
    palettes = ["Accent", "Accent_r", "Blues", "Blues_r", "BrBG", "BrBG_r", "BuGn", "BuGn_r", "BuPu", "BuPu_r",
                "CMRmap", "CMRmap_r", "Dark2", "Dark2_r", "GnBu", "GnBu_r", "Greens", "Greens_r", "Greys", "Greys_r",
                "OrRd", "OrRd_r", "Oranges", "Oranges_r", "PRGn", "PRGn_r", "Paired", "Paired_r", "Pastel1",
                "Pastel1_r", "Pastel2", "Pastel2_r", "PiYG", "PiYG_r", "PuBu", "PuBuGn", "PuBuGn_r", "PuBu_r", "PuOr",
                "PuOr_r", "PuRd", "PuRd_r", "Purples", "Purples_r", "RdBu", "RdBu_r", "RdGy", "RdGy_r", "RdPu",
                "RdPu_r", "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r", "Reds", "Reds_r", "Set1", "Set1_r", "Set2",
                "Set2_r", "Set3", "Set3_r", "Spectral", "Spectral_r", "Wistia", "Wistia_r", "YlGn", "YlGnBu",
                "YlGnBu_r", "YlGn_r", "YlOrBr", "YlOrBr_r", "YlOrRd", "YlOrRd_r", "afmhot", "afmhot_r", "autumn",
                "autumn_r", "binary", "binary_r", "bone", "bone_r", "brg", "brg_r", "bwr", "bwr_r", "cividis",
                "cividis_r", "cool", "cool_r", "coolwarm", "coolwarm_r", "copper", "copper_r", "cubehelix",
                "cubehelix_r", "flag", "flag_r", "gist_earth", "gist_earth_r", "gist_gray", "gist_gray_r", "gist_heat",
                "gist_heat_r", "gist_ncar", "gist_ncar_r", "gist_rainbow", "gist_rainbow_r", "gist_stern",
                "gist_stern_r", "gist_yarg", "gist_yarg_r", "gnuplot", "gnuplot2", "gnuplot2_r", "gnuplot_r", "gray",
                "gray_r", "hot", "hot_r", "hsv", "hsv_r", "icefire", "icefire_r", "inferno", "inferno_r", "magma",
                "magma_r", "mako", "mako_r", "nipy_spectral", "nipy_spectral_r", "ocean", "ocean_r", "pink", "pink_r",
                "plasma", "plasma_r", "prism", "prism_r", "rainbow", "rainbow_r", "rocket", "rocket_r", "seismic",
                "seismic_r", "spring", "spring_r", "summer", "summer_r", "tab10", "tab10_r", "tab20", "tab20_r",
                "tab20b", "tab20b_r", "tab20c", "tab20c_r", "terrain", "terrain_r", "twilight", "twilight_r",
                "twilight_shifted", "twilight_shifted_r", "viridis", "viridis_r", "vlag", "vlag_r", "winter",
                "winter_r"]

    import itertools
    return itertools.cycle(palettes)


def matplotlib_init():
    params = {
        'axes.labelsize': 16,
        'font.size': 16,
        'legend.fontsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.family': 'sans-serif'
    }
    plt.rcParams.update(params)
    colors = sns.color_palette("deep")
    return colors

def make_volcano(tab, lfc=0, FDR=0.05, title="", ylim=np.inf):
    sig = tab[(tab["FDR"]<FDR)&(tab["logFC"].abs()>lfc)]
    sns.scatterplot(x=tab["logFC"],y=-np.log10(tab["FDR"]), edgecolor=None, color="grey")
    sns.scatterplot(x=sig["logFC"],y=-np.log10(sig["FDR"]), edgecolor=None)
    plt.ylabel("-log10 FDR")
    plt.axhline(-np.log10(FDR),ls="--",color="red")
    if lfc > 0:
        plt.axvline(lfc,ls="--",color="red")
        plt.axvline(-lfc,ls="--",color="red")
    if ylim < np.inf:
        plt.ylim(-0.05*ylim, ylim)
    plt.title(f"{title} DEGs: {len(sig)}")
