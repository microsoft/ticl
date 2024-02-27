import pickle

import numpy as np

from mothernet import TabPFNClassifier
from mothernet.utils import get_mn_model


def test_follow_sklearn_interface():
    xs = np.random.rand(100, 99)
    ys = np.random.randint(0, 3, 100)

    eval_position = xs.shape[0] // 2
    train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
    test_xs, _ = xs[eval_position:], ys[eval_position:]

    classifier = TabPFNClassifier(device='cpu')
    classifier.fit(train_xs, train_ys)
    print(classifier)  # this might fail in some scenarios
    pred1 = classifier.predict_proba(test_xs)
    pickle_dump = pickle.dumps(classifier)
    classifier = pickle.loads(pickle_dump)
    pred2 = classifier.predict_proba(test_xs)
    assert (pred1 == pred2).all()


def test_our_tabpfn():
    xs = np.random.rand(100, 99)
    ys = np.random.randint(0, 3, 100)

    eval_position = xs.shape[0] // 2
    train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
    test_xs, _ = xs[eval_position:], ys[eval_position:]
    model_string = "tabpfn_nooptimizer_emsize_512_nlayers_12_steps_2048_bs_32ada_lr_0.0001_1_gpu_07_24_2023_01_43_33"
    epoch = "1650"
    get_mn_model(f"{model_string}_epoch_{epoch}.cpkt")
    classifier = TabPFNClassifier(device='cpu', model_string=model_string, epoch=epoch)
    classifier.fit(train_xs, train_ys)
    print(classifier)  # this might fail in some scenarios
    pred1 = classifier.predict_proba(test_xs)
    pickle_dump = pickle.dumps(classifier)
    classifier = pickle.loads(pickle_dump)
    pred2 = classifier.predict_proba(test_xs)
    assert (pred1 == pred2).all()