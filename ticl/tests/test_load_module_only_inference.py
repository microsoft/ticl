from ticl.datasets import load_openml_list, open_cc_dids
from ticl.prediction.tabpfn import TabPFNClassifier


def test_main():
    test_datasets, cc_test_datasets_multiclass_df = load_openml_list(
        open_cc_dids[:1], multiclass=True,
        shuffled=True,
        filter_for_nan=False,
        max_samples=10000,
        num_feats=100,
        return_capped=True,
        classification=True,
    )

    classifier_with_only_inference = TabPFNClassifier(device='cpu')
    classifier_normal = TabPFNClassifier(device='cpu')

    for dataset in test_datasets:
        xs, ys = dataset[1].clone(), dataset[2].clone()
        eval_position = xs.shape[0] // 2
        train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
        test_xs, _ = xs[eval_position:], ys[eval_position:]

        classifier_with_only_inference.fit(train_xs, train_ys)
        classifier_normal.fit(train_xs, train_ys)

        prediction_with_only_inference = classifier_with_only_inference.predict_proba(test_xs)
        prediction_normal = classifier_normal.predict_proba(test_xs)

        assert prediction_normal.shape == prediction_with_only_inference.shape
        number_of_predictions, number_of_classes = prediction_normal.shape

        for number in range(number_of_predictions):
            for class_nr in range(number_of_classes):
                # checks that every class probability has difference of at most
                assert (prediction_with_only_inference[number][class_nr] ==
                        prediction_normal[number][class_nr])
