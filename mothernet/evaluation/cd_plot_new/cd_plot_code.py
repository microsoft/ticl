"""Code for CD Plots.
From Code for AutoML Post Hoc Ensembling papers.

Requirements: the usual (pandas, matplotlib, numpy) and autorank==1.1.3
"""

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank._util import RankResult, get_sorted_rank_groups, rank_multiple_nonparametric, test_normality


def _custom_cd_diagram(result, reverse, ax, width, linewidth=0.7, linewidth_bar=2.5):
    """!TAKEN FROM AUTORANK WITH MODIFICATIONS!"""

    def plot_line(line, color="k", **kwargs):
        ax.plot([pos[0] / width for pos in line], [pos[1] / height for pos in line], color=color, **kwargs)

    def plot_text(x, y, s, *args, **kwargs):
        ax.text(x / width, y / height, s, *args, **kwargs)

    sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse)

    lowv = min(1, int(math.floor(min(sorted_ranks))))
    highv = max(len(sorted_ranks), int(math.ceil(max(sorted_ranks))))
    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        relative_rank = rank - lowv if not reverse else highv - rank
        return textspace + scalewidth / (highv - lowv) * relative_rank

    linesblank = 0.2 + 0.2 + (len(groups) - 1) * 0.1

    # add scale
    distanceh = 0.1
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((len(sorted_ranks) + 1) / 2) * 0.2 + minnotsignificant

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    plot_line([(textspace, cline), (width - textspace, cline)], linewidth=linewidth)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in [*list(np.arange(lowv, highv, 0.5)), highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2), (rankpos(a), cline)], linewidth=linewidth)

    for a in range(lowv, highv + 1):
        plot_text(rankpos(a), cline - tick / 2 - 0.05, str(a), ha="center", va="bottom")

    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline), (rankpos(sorted_ranks[i]), chei), (textspace - 0.1, chei)], linewidth=linewidth)
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline), (rankpos(sorted_ranks[i]), chei), (textspace + scalewidth + 0.1, chei)], linewidth=linewidth)
        plot_text(textspace + scalewidth + 0.2, chei, names[i], ha="left", va="center")

    # upper scale
    # if not reverse:
    #     begin, end = rankpos(lowv), rankpos(lowv + cd)
    # else:
    #     begin, end = rankpos(highv), rankpos(highv - cd)
    distanceh += 0.15
    bigtick /= 2
    # plot_line([(begin, distanceh), (end, distanceh)], linewidth=linewidth)
    # plot_line([(begin, distanceh + bigtick / 2), (begin, distanceh - bigtick / 2)], linewidth=linewidth)
    # plot_line([(end, distanceh + bigtick / 2), (end, distanceh - bigtick / 2)], linewidth=linewidth)
    # plot_text((begin + end) / 2, distanceh - 0.05, "CD", ha="center", va="bottom")

    # no-significance lines
    side = 0.05
    no_sig_height = 0.1
    start = cline + 0.2
    for l, r in groups:
        plot_line([(rankpos(sorted_ranks[l]) - side, start), (rankpos(sorted_ranks[r]) + side, start)], linewidth=linewidth_bar)
        start += no_sig_height

    return ax


def cd_evaluation(performance_per_dataset, maximize_metric, verbose=False, ignore_non_significance=False, ax=None, linewidth=1, linewidth_bar=2):
    """Performance per dataset is  a dataframe that stores the performance (with respect to a metric) for  set of
    configurations / models / algorithms per dataset. In  detail, the columns are individual configurations.
    rows are datasets and a cell is the performance of the configuration for  dataset.
    """
    # -- Preprocess data for autorank
    rank_data = performance_per_dataset.copy() * -1 if maximize_metric else performance_per_dataset.copy()
    rank_data = rank_data.reset_index(drop=True)
    rank_data = pd.DataFrame(rank_data.values, columns=list(rank_data))

    # -- Settings for autorank
    alpha = 0.05
    effect_size = None
    verbose = verbose
    order = "ascending"  # always due to the preprocessing
    alpha_normality = alpha / len(rank_data.columns)
    all_normal, pvals_shapiro = test_normality(rank_data, alpha_normality, verbose)

    # -- Friedman-Nemenyi
    res = rank_multiple_nonparametric(rank_data, alpha, verbose, all_normal, order, effect_size, None)

    result = RankResult(
        res.rankdf,
        res.pvalue,
        res.cd,
        res.omnibus,
        res.posthoc,
        all_normal,
        pvals_shapiro,
        None,
        None,
        None,
        alpha,
        alpha_normality,
        len(rank_data),
        None,
        None,
        None,
        None,
        res.effect_size,
        None,
    )
    # print(res.rankdf)
    if result.pvalue >= result.alpha:
        if ignore_non_significance:
            warnings.warn("Result is not significant and results of the plot may be misleading!")
        else:
            raise ValueError(
                "Result is not significant and results of the plot may be misleading. If you still want to see the CD plot, set"
                + " ignore_non_significance to True.",
            )

    # -- Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
    plt.rcParams.update({"font.size": 16})
    _custom_cd_diagram(result, order == "ascending", ax, 8, linewidth=linewidth, linewidth_bar=linewidth_bar)

    return result
