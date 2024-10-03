from itertools import cycle
import os

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


def exp_curve(x, params):
    if len(params) == 2:
        a, b, c = *params, 0
    elif len(params) == 3:
        a, b, c = params
    else:
        raise ValueError(f"Invalid parameter length: {len(params)}")

    return a * (x + 1e-5) ** b + c


def fit_exp_curve(x, y, include_offset=False, alpha=0):
    def exp_loss(params):
        if include_offset:
            return ((exp_curve(x, params) - y) ** 2).mean() + alpha * params[2] ** 2
        else:
            return ((exp_curve(x, params) - y) ** 2).mean()
    if include_offset:
        return minimize(exp_loss, [1, -1, 0], bounds=[(0, 100), (-10, -1e-10), (0, 100)])
    else:
        return minimize(exp_loss, [1, -1], bounds=[(0, 100), (-10, -1e-10)])


class ExponentialRegression(RegressorMixin, BaseEstimator):
    def __init__(self, include_offset=False, alpha=0):
        self.include_offset = include_offset
        self.alpha = alpha

    def fit(self, X, y):
        assert X.ndim == 1 or X.shape[1] == 1
        X = np.array(X).ravel()
        self.optimization_result_ = fit_exp_curve(X, y, include_offset=self.include_offset, alpha=self.alpha)
        self.fitted_func_ = lambda x: exp_curve(x, self.optimization_result_.x)
        return self

    def predict(self, X):
        return self.fitted_func_(np.array(X).ravel())


def plot_exponential_regression(loss_df, x='epoch', y='loss', hue='run', extrapolation_factor=3, verbose=0, er=None):
    import plotly.graph_objects as go

    fig = go.Figure()
    if er is None:
        er = ExponentialRegression()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692',
              '#B6E880', '#FF97FF', '#FECB52']

    for color, run in zip(cycle(colors), loss_df[hue].unique()):
        this_df = loss_df[loss_df[hue] == run]
        this_X = this_df[[x]]
        this_y = this_df[y]
        er.fit(this_X, this_y)
        if verbose:
            print(run)
            print(er.score(this_X, this_y))
        pred_train = er.predict(this_X)
        # start of extrapolation is per-run, end is the same for all runs
        extrapolate = np.linspace(this_X[x].max(), loss_df[x].max() * extrapolation_factor, num=100)
        pred_extrapolation = er.predict(extrapolate.reshape(-1, 1))
        fig.add_trace(go.Scatter(x=this_X[x], y=this_y, mode="markers", name=run, marker_color=color,
                      opacity=.3, legendgroup=run, showlegend=False, hoverinfo="name", hoverlabel_namelength=-1))
        fig.add_trace(go.Scatter(x=this_X[x], y=pred_train, mode="lines", name=run, marker_color=color,
                      legendgroup=run, showlegend=True, hoverinfo="name", hoverlabel_namelength=-1))
        fig.add_trace(go.Scatter(x=extrapolate, y=pred_extrapolation, mode="lines", name=run, marker_color=color,
                      line_dash="dash", legendgroup=run, showlegend=False, hoverinfo="name", hoverlabel_namelength=-1))
    fig.update_layout(xaxis_title=x, yaxis_title=y, height=800)
    return fig


def plot_exponential_smoothing(loss_df, x='time_days', y='loss', hue='run', extra_smoothing=1, logx=True, logy=True, inactive_legend=False):
    import plotly.graph_objects as go
    fig = go.Figure()
    for run in loss_df[hue].unique():
        this_df = loss_df[loss_df[hue] == run]
        if extra_smoothing == 0:
            smoothed = this_df[[y, x]].reset_index()
        else:
            try:
                smoothed = this_df[[y, x]].ewm(span=len(this_df) / this_df[x].max() / 2 * extra_smoothing).mean().reset_index()
            except ValueError as e:
                print(e)
                continue
        if 'status' in this_df.columns and (this_df.status != "RUNNING").all():
            fig.add_trace(go.Scatter(x=smoothed[x], y=smoothed[y], mode='lines', name=run, hoverinfo="name",
                          hoverlabel_namelength=-1, opacity=.3, showlegend=inactive_legend))
        else:
            fig.add_trace(go.Scatter(x=smoothed[x], y=smoothed[y], mode='lines', name=run, hoverinfo="name", hoverlabel_namelength=-1))
    fig.update_layout(height=1200)
    if logx:
        fig.update_xaxes(type="log")
    if logy:
        fig.update_yaxes(type="log")
    fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
    return fig


def get_runs(filter_string, experiment_id):
    return MlflowClient().search_runs(
        experiment_ids=experiment_id, filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY, order_by=["metrics.accuracy DESC"])


def plot_experiment(experiment_name=None, experiment_id=None, x="epoch", y="loss", verbose=False, logx=True, logy=True, return_df=False, extra_smoothing=1,
                    filter_runs=("running", "reference"), mlflow_host=None, legend=False, inactive_legend=False, filter_like=None):
    if mlflow_host is None:
        mlflow_host = os.environ.get("MLFLOW_HOSTNAME", None)
    if mlflow_host is None:
        raise ValueError("Please specify mlflow_host or set MLFLOW_HOSTNAME environment variable.")
    mlflow.set_tracking_uri(f"http://{mlflow_host}:5000")
    if experiment_name is not None and experiment_id is not None:
        raise ValueError("Please specify either experiment_name or experiment_id, not both.")
    if experiment_name is not None:
        experiment_id = MlflowClient().get_experiment_by_name(experiment_name).experiment_id
    else:
        experiment_id = experiment_id or "0"

    if filter_like is not None:
        filter_string = f"attributes.run_name LIKE '{filter_like}'"
        if filter_runs != "all":
            filter_string += " AND "
    else:
        filter_string = ""
    runs = []
    if filter_runs == "all":
        runs = get_runs(filter_string, experiment_id)
    else:
        if "running" in filter_runs:
            runs.extend(get_runs("attribute.status='RUNNING'" + filter_string, experiment_id))
        if "reference" in filter_runs:
            runs.extend(get_runs('tags.reference = "True"' + filter_string, experiment_id))

    losses_all = []
    already_seen = set()
    for run in runs:
        if run.info.run_id in already_seen:
            continue
        already_seen.add(run.info.run_id)
        try:
            losses = MlflowClient().get_metric_history(run.info.run_id, key=y)
            if not len(losses):
                continue
            adjusted_wallclock = MlflowClient().get_metric_history(run.info.run_id, key="wallclock_time")
            losses_df = pd.DataFrame.from_dict([dict(l) for l in losses]).rename(columns={'value': y})
            clock_df = pd.DataFrame.from_dict([dict(t) for t in adjusted_wallclock]).rename(columns={'value': 'clock'})
            losses_df = losses_df.merge(clock_df[['step', 'clock']], on='step')
            losses_df['run'] = run.info.run_name
            losses_df['timestamp'] = losses_df.timestamp / 1000 / (60 * 60 * 24)
            losses_df['time_days'] = losses_df['clock'] / (60 * 60 * 24)
            losses_df['status'] = run.info.status
            losses_all.append(losses_df)
        except MlflowException as e:
            if verbose:
                print(e)
    losses_all_df = pd.concat(losses_all, ignore_index=True).rename(columns={'step': 'epoch'})
    losses_all_df['timestamp'] -= losses_all_df.timestamp.min()
    fig = plot_exponential_smoothing(losses_all_df, x=x, y=y, logx=logx, logy=logy, extra_smoothing=extra_smoothing, inactive_legend=inactive_legend)
    if legend:
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
    else:
        fig.update_layout(showlegend=False)
    if return_df:
        return fig, losses_all_df
    return fig
