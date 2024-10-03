import pandas as pd

from ticl.plot_code.cd_plot_code import cd_evaluation
from ticl.plot_code.ni_plot_code import ni_evaluation


def eval_am():
    input_data = pd.read_csv("res/results_test.csv", index_col=0)

    # filter down to max time
    input_data = input_data[(input_data["max_time"] == 3600) | (input_data["best"].isna())]

    # Take mean over folds
    input_data = input_data.groupby(["dataset", "model"]).mean(numeric_only=True).reset_index()

    # Pivot for desired metric to create the performance per dataset table
    performance_per_dataset = input_data.pivot(index="dataset", columns="model", values="mean_metric")

    cd_evaluation(performance_per_dataset, maximize_metric=True, output_path=None, ignore_non_significance=False)
    ni_evaluation(performance_per_dataset, maximize_metric=True, baseline_method="KNN")


if __name__ == "__main__":
    eval_am()
