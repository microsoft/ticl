from sklearn.base import BaseEstimator, RegressorMixin

from scipy.optimize import minimize

def exp_curve(x, a, b, c):
    return a * (x + 1) ** b + c

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