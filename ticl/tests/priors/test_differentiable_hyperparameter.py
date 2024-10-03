from ticl.distributions import parse_distribution as DifferentiableHyperparameter
import lightning as L


def test_differentiable_hyper_uniform():
    L.seed_everything(42)
    hyperparameter = DifferentiableHyperparameter(name='test', distribution='uniform', min=0.0, max=1.0)
    samples = [hyperparameter() for _ in range(11)]
    assert samples[0] == 0.3745401188473625
    assert samples[10] == 0.020584494295802447


def test_differentiable_hyper_meta_beta():
    L.seed_everything(42)
    hyperparameter = DifferentiableHyperparameter(name='test', distribution='meta_beta', scale=0.6, min=0.1, max=5.0)
    samples = [hyperparameter() for _ in range(11)]
    assert samples[0]() == 0.15255399122616758
    assert samples[0]() == 0.2707540740311954
    assert samples[10]() == 0.17798037863404909
    assert samples[10]() == 0.47444729678154285


def test_differentiable_hyper_meta_gamma():
    L.seed_everything(42)
    hyperparameter = DifferentiableHyperparameter(name='test', distribution='meta_gamma', max_alpha=2, max_scale=100, round=True, lower_bound=4)
    samples = [hyperparameter() for _ in range(11)]
    assert samples[0]() == 27
    assert samples[0]() == 14
    assert samples[10]() == 16
    assert samples[10]() == 6


def test_differentiable_hyper_meta_trunc_norm_log_scaled():
    L.seed_everything(42)
    hyperparameter = DifferentiableHyperparameter(name='test', distribution='meta_trunc_norm_log_scaled',
                                                  max_mean=.3, min_mean=0.0001, round=False, lower_bound=0.0)
    samples = [hyperparameter() for _ in range(11)]
    assert samples[0]() == 0.0014598013986488975
    assert samples[0]() == 0.001735194550030197
    assert samples[10]() == 0.013383656409053687
    assert samples[10]() == 0.013613155757117055


def test_differentiable_hyper_meta_meta_choice():
    L.seed_everything(42)
    hyperparameter = DifferentiableHyperparameter(name='test', distribution='meta_choice', choice_values=[0.00001, 0.0001, 0.01])
    samples = [hyperparameter() for _ in range(11)]
    assert samples[0] == 0.01
    assert samples[10] == 0.0001


def test_differentiable_hyper_meta_meta_choice_mixed():
    L.seed_everything(42)
    hyperparameter = DifferentiableHyperparameter(name='test', distribution='meta_choice_mixed', choice_values=[lambda: "a", lambda: "b", lambda: "c"])
    samples = [hyperparameter() for _ in range(11)]
    assert samples[0]()() == "c"
    assert samples[0]()() == "c"
    assert samples[10]()() == "a"
    assert samples[10]()() == "b"
