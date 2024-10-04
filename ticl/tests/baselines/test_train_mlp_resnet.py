from ticl.evaluation.baselines.torch_mlp import TorchMLP
from ticl.evaluation.baselines.resnet import ResNetClassifier
from ticl.testing_utils import check_predict_iris


def test_train_mlp():
    # maybe adding a StandardScaler would be nice
    check_predict_iris(TorchMLP(device="cpu", n_epochs=30), check_accuracy=True)


def test_train_resnet():
    # maybe adding a StandardScaler would be nice
    check_predict_iris(ResNetClassifier(device="cpu", n_epochs=30), check_accuracy=True)