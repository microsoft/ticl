import numpy as np
import pytest

from mothernet.prediction.mothernet_additive import compute_top_pairs, MotherNetAdditiveClassifierPairEffects
from mothernet.utils import get_mn_model

np.random.seed(0)
n = 200
d = 10
X = np.random.rand(n, d)
# continuous xor with X0 and X3: x0 + x3 - 2 * x0 * x3
# => requires pairwise effect and top pair should be (0, 3)
y = X[:, 0] + X[:, 3] - 2 * X[:, 0] * X[:, 3]
y = y > 0.5


@pytest.mark.parametrize("n_pair_feature_max_ratio", [0, 0.5, 1.0])
@pytest.mark.parametrize("pair_strategy", ["sum_importance", "fast"])
def test_pair(n_pair_feature_max_ratio: float, pair_strategy: str):
    pairs = compute_top_pairs(X, y, pair_strategy=pair_strategy, n_pair_feature_max_ratio=n_pair_feature_max_ratio)

    assert len(pairs) == int(n_pair_feature_max_ratio * d)
    if pairs:
        assert pairs[0] == (0, 3)


@pytest.mark.parametrize("n_pair_feature_max_ratio", [0, 0.1])
@pytest.mark.parametrize("pair_strategy", ["sum_importance", "fast"])
def test_estimator(n_pair_feature_max_ratio: float, pair_strategy: str):
    baam_model_string = "baam_nsamples500_numfeatures10_04_07_2024_17_04_53_epoch_1780.cpkt"
    clf = MotherNetAdditiveClassifierPairEffects(
        path=get_mn_model(baam_model_string),
        pair_strategy=pair_strategy,
        n_pair_feature_max_ratio=n_pair_feature_max_ratio,
    )
    y_pred = clf.fit(X, y).predict(X)
    if n_pair_feature_max_ratio > 0:
        assert (y_pred == y).mean() > 0.9
