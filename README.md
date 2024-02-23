# MotherNet

The MotherNet is a hypernetwork foundational model (or conditional neural process) for tabular data classification.
Both the architecture and the code is based on the [TabPFN](https://github.com/automl/TabPFN) by the [Freiburg AutoML group](https://www.automl.org/).

This is a research prototype, shared for research use, and not meant for real-world applications.

## Installation

It's recommended to use conda to create an environment using the provided environment file:

```
conda create -f environment.yml
```

## Getting started

A simple usage of our sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))
```

### MotherNet Usage

MotherNet uses the same preprocessing as the TabPFN work it builds upon, but we found that using one-hot-encoding during inference improves accuracy.
Scaling of features is handled internally.

### Model Training
Full model training code is provided. Training ``MotherNet`` is possible with ``python fit_model.py``. A GPU ID can be specified with ``-g GPU_ID``. See the help for more options.
The results in the paper correspond to ``python fit_model.py -L 2``, though default values might change and no longer reflect the values in the paper.
Data-parallel Multi-GPU training is in principal supported using ``torchrun``.
By default, experiments are tracked using MLFlow if the ``MLFLOW_HOSTNAME`` environment variable is set. Using MLFlow for a particular run can be disabled with the ``--no-mlflow`` argument.

## Papers
This work is described in [MotherNet: A Foundational Hypernetwork for Tabular Classification](https://arxiv.org/pdf/2312.08598).
Please cite that work when using this code. As this work rests on the TabPFN work, I would suggest you also cite their [paper](https://arxiv.org/abs/2207.01848),
which also provides more background on the methodology.

## License
Copyright 2022 Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter

Additions by Andreas Mueller, 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
