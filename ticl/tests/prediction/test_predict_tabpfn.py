import pickle

import numpy as np

from ticl.prediction import TabPFNClassifier
from sklearn.model_selection import train_test_split


def test_many_classes():
    # test that if more than 10 classes, least frequent classes are put into bucket (class 9 here)
    classes = np.array(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])
    rng = np.random.RandomState(42)
    xs = rng.uniform(size=(400, 2))
    ys = np.digitize(xs[:, 0], bins=np.linspace(0, 1, 11)) - 1  # 0-9
    ys_more_classes = ys.copy()
    ys_more_classes[ys > 9] = rng.randint(10, 20, size=(ys > 9).sum())
    ys_more_classes_str = classes[ys_more_classes]
    X_train, X_test, y_train, y_test, y_org_train, y_org_test = train_test_split(xs, ys_more_classes_str, ys, random_state=42)

    classifier = TabPFNClassifier(device='cpu')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    mask = y_org_test < 9
    assert (y_pred[mask] == y_test[mask]).mean() > 0.90
    # should be "nine" which is the biggest class of remainder
    assert (y_pred[~mask] == 'nine').all()