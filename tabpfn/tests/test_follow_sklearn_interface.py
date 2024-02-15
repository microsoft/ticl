import pickle

import numpy as np

from tabpfn import TabPFNClassifier


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
