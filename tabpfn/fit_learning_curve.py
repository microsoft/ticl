from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from itertools import cycle

from scipy.optimize import minimize

def exp_curve(x, a, b, c):
    return a * (x + 1e-5) ** b + c

def fit_exp_curve(x, y):
    def exp_loss(params):
        return ((exp_curve(x, params[0], params[1], params[2]) - y) ** 2).mean()
    return minimize(exp_loss, [1, -1, 0], bounds=[(0, 100), (-10, -1e-10), (0, 100)])
    
def make_pred_func(res):
    return lambda x: exp_curve(x, res.x[0], res.x[1], res.x[2])
    

class ExponentialRegression(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        assert X.ndim == 1 or X.shape[1] == 1
        X = X.ravel()
        self.optimization_result_ = fit_exp_curve(X, y)
        self.fitted_func_ = make_pred_func(self.optimization_result_)
        return self
        
    def predict(self, X):
        return self.fitted_func_(X)

import plotly.graph_objects as go
from tabpfn.fit_learning_curve import ExponentialRegression


def plot_exponential_regression(loss_df, x='epoch', y='loss', hue='run', extrapolation_factor=3, verbose=0):
    import plotly.graph_objects as go

    fig = go.Figure()

    er = ExponentialRegression()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692',
              '#B6E880', '#FF97FF', '#FECB52']

    for color, run in zip(cycle(colors), loss_df[hue].unique()):
        this_df = loss_df[loss_df[hue] == run]
        this_X = this_df[x]
        this_y = this_df[y]
        er.fit(this_X, this_y)
        if verbose:
            print(run)
            print(er.score(this_X, this_y))
        pred_train = er.predict(this_X)
        # start of extrapolation is per-run, end is the same for all runs
        extrapolate = np.linspace(this_X.max(), loss_df[x].max() * extrapolation_factor, num=100)
        pred_extrapolation = er.predict(extrapolate)

        fig.add_trace(go.Scatter(x=this_X, y=this_y, mode="markers", name=run, marker_color=color, opacity=.3, legendgroup=run, showlegend=False))
        fig.add_trace(go.Scatter(x=this_X, y=pred_train, mode="lines", name=run, marker_color=color, legendgroup=run, showlegend=True))
        fig.add_trace(go.Scatter(x=extrapolate, y=pred_extrapolation, mode="lines", name=run, marker_color=color, line_dash="dash", legendgroup=run, showlegend=False))
    fig.update_layout(xaxis_title=x, yaxis_title=y, height=800)
    return fig