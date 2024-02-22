"""Code for Normalized Improvement Plot
From Code for AutoML Post Hoc Ensembling papers.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from matplotlib.lines import Line2D


def normalize_performance(ppd, baseline_algorithm, higher_is_better):
    # I know this is inefficient and I know a better way to implement it just using pandas and no apply, but I am too lazy to do it now.
    def normalize_function(row):
        # https://stats.stackexchange.com/a/178629 scale to -1 = performance of baseline to 0 = best performance

        tmp_row = row.copy() * -1 if not higher_is_better else row.copy()

        baseline_performance = tmp_row[baseline_algorithm]
        range_fallback = abs(tmp_row.max() - baseline_performance) == 0

        if range_fallback:
            mask = abs(tmp_row - baseline_performance) == 0
            tmp_row[~mask] = -10  # par10 like

            tmp_row[mask] = -1
            return tmp_row

        return (tmp_row - baseline_performance) / (tmp_row.max() - baseline_performance) - 1

    return ppd.apply(normalize_function, axis=1) + 1


def normalized_improvement_boxplot(normalized_ppd, baseline_algorithm, m_name="model"):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = pd.melt(normalized_ppd.drop(baseline_algorithm, axis=1))
    xlim = min(list(plot_df.groupby(m_name).apply(lambda x: boxplot_stats(x).pop(0)["whislo"]))) - 0.5
    outlier = plot_df.groupby(m_name).apply(lambda x: sum(boxplot_stats(x).pop(0)["fliers"] < xlim)).to_dict()
    if len(list(normalized_ppd)) > len(sns.color_palette("tab10")):
        # catch edge case
        palette = None
    else:
        palette = {k: sns.color_palette("tab10")[i] for i, k in enumerate(sorted(normalized_ppd))}
    sns.boxplot(data=plot_df, y=m_name, x="value", palette=palette, showfliers=False)
    sns.stripplot(data=plot_df, y=m_name, x="value", color="black")
    ax.axvline(x=0, c="red")
    plt.xlabel("Normalized Improvement (Higher is better)")
    yticks = [item.get_text() for item in ax.get_yticklabels()]
    new_yticks = [ytick + f" [{outlier[ytick]}]" for ytick in yticks]
    ax.set_yticklabels(new_yticks)
    plt.xlim(xlim, 1)
    plt.legend(handles=[Line2D([0], [0], label=baseline_algorithm, color="r")])
    plt.ylabel("Method")
    plt.tight_layout()
    plt.show()
    plt.close()


def ni_evaluation(performance_per_dataset, maximize_metric, baseline_method, m_name="model"):
    normalized_perf_pd = normalize_performance(performance_per_dataset, baseline_method, maximize_metric)
    normalized_improvement_boxplot(normalized_perf_pd, baseline_method, m_name=m_name)
